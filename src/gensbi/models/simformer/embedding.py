import jax
from jax import numpy as jnp
from flax import nnx
import numpy as np
from jax.typing import DTypeLike 
from jax import Array

class MLPEmbedder(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.p_skip = nnx.Param(0.01*jnp.ones((1, 1, hidden_dim)))
        self.in_layer = nnx.Linear(
            in_features=in_dim,
            out_features=hidden_dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.silu = nnx.silu
        self.out_layer = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        x = jnp.atleast_1d(x)
        out =  self.out_layer(self.silu(self.in_layer(x)))
        x_repeated, out = jnp.broadcast_arrays(x, out)
        out = x_repeated * self.p_skip + (1-self.p_skip)*out
        return out

class SimpleTimeEmbedding(nnx.Module):
    def __init__(self):
        """Simple time embedding module. Mostly used to embed time.

        """
        return 
    def __call__(self, t):
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            t = jnp.expand_dims(t, axis=1)
        out = jnp.concatenate([
            t - 0.5,
            jnp.cos(2 * jnp.pi * t),
            jnp.sin(2 * jnp.pi * t),
            -jnp.cos(4 * jnp.pi * t)
        ], axis=-1)
        return out


class SinusoidalEmbedding(nnx.Module):
    def __init__(self, output_dim: int = 128):
        """Sinusoidal embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
        """
        self.output_dim = output_dim
        return

    def __call__(self, t):
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            t = jnp.expand_dims(t, axis=1)
        half_dim = self.output_dim // 2 + 1
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = jnp.expand_dims(emb, 0)
        # emb = t[..., None] * emb[None, ...]
        emb = jnp.dot(t, emb)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb[..., : self.output_dim]




# class GaussianFourierEmbedding(nnx.Module):
#     def __init__(
#         self,
#         output_dim: int = 128,
#         learnable: bool = False,
#         *,
#         rngs: nnx.Rngs
#     ):
#         """Gaussian Fourier embedding module. Mostly used to embed time.

#         Args:
#             output_dim (int, optional): Output dimesion. Defaults to 128.
#         """
#         self.output_dim = output_dim
#         self.B = nnx.Param(jax.random.normal(rngs.params(), [self.output_dim // 2 + 1]))
#         self.learnable = learnable
#         return

        
#     def __call__(self, t):
#         t = jnp.atleast_1d(t)
#         if t.ndim == 1:
#             t = jnp.expand_dims(t, axis=1)

#         if not self.learnable:
#             B = jnp.expand_dims(jax.lax.stop_gradient(self.B), 0)
#         else:
#             B = jnp.expand_dims(self.B, 0)

#         arg = 2 * jnp.pi * jnp.dot(t,B)
#         term1 = jnp.cos(arg)
#         term2 = jnp.sin(arg)
#         out = jnp.concatenate([term1, term2], axis=-1)
#         return out[..., : self.output_dim]



class GaussianFourierEmbedding(nnx.Module):
    def __init__(
        self,
        output_dim: int = 128,
        learnable: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
        """
        self.output_dim = output_dim
        half_dim = self.output_dim // 2 + 1
        self.B = nnx.Param(jax.random.normal(rngs.params(), [half_dim , 1]))
        if not learnable:
            self.B = jax.lax.stop_gradient(jnp.array(self.B))
   
        return

        
    def __call__(self, t):
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            t = jnp.expand_dims(t, axis=1)

        # B = jax.lax.cond(
        #     self.learnable,
        #     lambda: self.B,  # True branch: use B directly
        #     lambda: jax.lax.stop_gradient(self.B)  # False branch: use B with stop_gradient
        # )
        B = self.B

        arg = 2 * jnp.pi * jnp.dot(t,B.T)
        term1 = jnp.cos(arg)
        term2 = jnp.sin(arg)
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]