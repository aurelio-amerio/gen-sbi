from dataclasses import dataclass

from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import nnx
from jax.typing import DTypeLike

from einops import repeat, rearrange

from gensbi.models.flux1.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

from gensbi.utils.model_wrapping import ModelWrapper


@dataclass
class FluxParams:
    """Parameters for the Flux model.
    
    Args:
        in_channels (int): Number of input channels.
        vec_in_dim (Union[int, None]): Dimension of the vector input, if applicable.
        context_in_dim (int): Dimension of the context input.
        mlp_ratio (float): Ratio for the MLP layers.
        num_heads (int): Number of attention heads.
        depth (int): Number of double stream blocks.
        depth_single_blocks (int): Number of single stream blocks.
        axes_dim (list[int]): Dimensions of the axes for positional encoding.
        qkv_bias (bool): Whether to use bias in QKV layers.
        rngs (nnx.Rngs): Random number generators for initialization.
        obs_dim (int): Observation dimension.
        cond_dim (int): Condition dimension.
        use_rope (bool): Whether to use Rotary Position Embedding (RoPE).
        theta (int): Scaling factor for positional encoding.
        guidance_embed (bool): Whether to use guidance embedding.
        qkv_multiplier (int): Multiplier for QKV features.
        param_dtype (DTypeLike): Data type for model parameters.
        
    """
    in_channels: int
    vec_in_dim: Union[int, None]
    context_in_dim: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    qkv_bias: bool
    rngs: nnx.Rngs
    obs_dim: int   # observation dimension
    cond_dim: int  # condition dimension
    use_rope: bool = True
    theta: int = 10_000
    guidance_embed: bool = False
    qkv_multiplier: int = 1
    param_dtype: DTypeLike = jnp.bfloat16

    def __post_init__(self):
        self.hidden_size = int(jnp.sum(jnp.asarray(self.axes_dim, dtype=jnp.int32)) * self.qkv_multiplier * self.num_heads)
        self.qkv_features = self.hidden_size // self.qkv_multiplier


class Identity(nnx.Module):
    def __call__(self, x: Array) -> Array:
        return x


class Flux(nnx.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.qkv_features = params.qkv_features

        pe_dim = self.qkv_features // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.obs_in = nnx.Linear(
            in_features=self.in_channels,
            out_features=self.hidden_size,
            use_bias=True,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )
        self.time_in = MLPEmbedder(
            in_dim=256,
            hidden_dim=self.hidden_size,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )
        self.vector_in = (MLPEmbedder(
            params.vec_in_dim,
            self.hidden_size,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        ) if params.guidance_embed else Identity())
    
        self.cond_in = nnx.Linear(
            in_features=params.context_in_dim,
            out_features=self.hidden_size,
            use_bias=True,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )

        self.condition_embedding = nnx.Param(0.01 * jnp.ones((1, self.hidden_size), dtype=params.param_dtype))
        self.condition_null = nnx.Param(jax.random.normal(params.rngs.cond(), (1,params.cond_dim, self.hidden_size), dtype=params.param_dtype))

        self.double_blocks = nnx.Sequential(
            *[
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_features=self.qkv_features,
                    qkv_bias=params.qkv_bias,
                    rngs=params.rngs,
                    param_dtype=params.param_dtype,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nnx.Sequential(
            *[
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_features=self.qkv_features,
                    rngs=params.rngs,
                    param_dtype=params.param_dtype,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            1,
            self.out_channels,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )

        self.use_rope = params.use_rope

        if not self.use_rope:
            # assert params.obs_dim is not None and params.cond_dim is not None, \
            #     "If not using RoPE, obs_dim and cond_dim must be specified."
            
            self.id_embedder = nnx.Embed(
                num_embeddings=params.obs_dim + params.cond_dim,
                features=self.hidden_size,
                rngs=params.rngs,
                param_dtype=params.param_dtype)
        else:
            self.id_embedder = Identity()

    def __call__(
        self,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        timesteps: Array,
        conditioned: bool | Array = True,
        guidance: Array | None = None,
    ) -> Array:
        
        obs = jnp.asarray(obs, dtype=self.params.param_dtype)  
        cond = jnp.asarray(cond, dtype=self.params.param_dtype)  
        timesteps = jnp.asarray(timesteps, dtype=self.params.param_dtype)  

        if obs.ndim != 3 or cond.ndim != 3:
            raise ValueError("Input obs and cond tensors must have 3 dimensions.")

        # running on sequences obs
        obs = self.obs_in(obs)
        vec = self.time_in(timestep_embedding(timesteps, 256))

        conditioned = jnp.asarray(conditioned, dtype=jnp.bool_)  # type: ignore
        conditioned_int = jnp.asarray(conditioned, dtype=jnp.int32)[...,None] # type: ignore

        condition_embedding = (
            self.condition_embedding * (1-conditioned_int)
        )

        vec = vec + condition_embedding # we add the condition embedding to the vector

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.vector_in(guidance)

        cond_processed = self.cond_in(cond)  # (B, F, H)
        cond_null = repeat(self.condition_null.value, '1 h c -> b h c', b=obs.shape[0])  # type: ignore
        cond = jnp.where(conditioned[...,None,None], cond_processed, cond_null)  # we replace the condition with a null vector if not conditioned

        ids = jnp.concatenate((cond_ids, obs_ids), axis=1)
        if self.use_rope:
            pe = self.pe_embedder(ids)
        else:
            ids = jnp.squeeze(ids, axis=-1) # ids should have dimension (B, F, 1)

            id_emb = self.id_embedder(ids)

            id_emb = jnp.broadcast_to(
            id_emb, (obs.shape[0], self.params.obs_dim + self.params.cond_dim, self.hidden_size)  # type: ignore
        )

            cond_ids_emb = id_emb[:, :cond.shape[1], :]
            obs_ids_emb = id_emb[:, cond.shape[1]:, :]
            
            obs = obs + obs_ids_emb
            cond = cond + cond_ids_emb

            pe=None

        for block in self.double_blocks.layers:
            obs, cond = block(obs=obs, cond=cond, vec=vec, pe=pe)

        obs = jnp.concatenate((cond, obs), axis=1)
        for block in self.single_blocks.layers:
            obs = block(obs, vec=vec, pe=pe)
        obs = obs[:, cond.shape[1] :, ...]

        obs = self.final_layer(obs, vec)  # (N, T, patch_size ** 2 * out_channels)
        return obs

class FluxWrapper(ModelWrapper):
    def __init__(self, model):
        super().__init__(model)

    def _call_model(self, x, t, args, **kwargs):
        x = jnp.atleast_1d(x)
        t = jnp.atleast_1d(t)

        if x.ndim < 3:
            x = rearrange(x, '... -> 1 ... 1' if x.ndim == 1 else '... -> ... 1')

        return jnp.squeeze(self.model(obs=x, timesteps=t, conditioned=True, **kwargs), axis=-1)