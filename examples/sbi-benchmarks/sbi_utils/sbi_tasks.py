
import jax
import jax.numpy as jnp

from functools import partial

from abc import ABC, abstractmethod

from .graph import faithfull_mask, min_faithfull_mask, moralize


from sbibm import get_task as _get_torch_task

import torch
import jax
import jax.numpy as jnp


class Task(ABC):
    
    def __init__(self, name: str, backend: str = "torch") -> None:
        self.name = name
        self.backend = backend    
        
    @property
    def theta_dim(self):
        return self.get_theta_dim()
    
    @property
    def x_dim(self):
        return self.get_x_dim()
    
    def get_theta_dim(self):
        raise NotImplementedError()
    
    def get_x_dim(self):
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_node_id(self):
        raise NotImplementedError()
    
    def get_base_mask_fn(self):
        raise NotImplementedError()
    
    

    
    
class InferenceTask(Task):
    
    observations = range(1, 11)
    
    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        
    def get_prior(self):
        raise NotImplementedError()
        
    def get_simulator(self):
        raise NotImplementedError()
    
    def get_data(self, num_samples: int, rng=None):
        raise NotImplementedError()
    
    def get_observation(self, index: int):
        raise NotImplementedError()
    
    def get_reference_posterior_samples(self, index: int):
        raise NotImplementedError()
    
    def get_true_parameters(self, index: int):
        raise NotImplementedError()
    
    def get_edge_mask_fn(self, name="undirected"):
        task = self.task
        if name.lower() == "faithfull":
            base_mask_fn = self.get_base_mask_fn()
            def faithfull_edge_mask(node_id, condition_mask, meta_data=None):
                base_mask = base_mask_fn(node_id, meta_data)
                return faithfull_mask(base_mask, condition_mask)

            return faithfull_edge_mask
        elif name.lower() == "min_faithfull":
            base_mask_fn = self.get_base_mask_fn()        
            def min_faithfull_edge_mask(node_id, condition_mask,meta_data=None):
                base_mask = base_mask_fn(node_id, meta_data)

                return min_faithfull_mask(base_mask, condition_mask)

            return min_faithfull_edge_mask
        elif name.lower() == "undirected":
            base_mask_fn = self.get_base_mask_fn()        
            def undirected_edge_mask(node_id, condition_mask, meta_data=None):
                base_mask = base_mask_fn(node_id, meta_data)
                return moralize(base_mask)
            
            return undirected_edge_mask
        
        elif name.lower() == "directed":
            base_mask_fn = self.get_base_mask_fn()        
            def directed_edge_mask(node_id, condition_mask, meta_data=None):
                base_mask = base_mask_fn(node_id, meta_data)
                return base_mask

            return directed_edge_mask
        elif name.lower() == "none":
            return lambda node_id, condition_mask, *args, **kwargs: None
        else:
            raise NotImplementedError()


class SBIBMTask(InferenceTask):
    observations = range(1, 11)

    def __init__(self, name: str, backend: str = "jax") -> None:
        super().__init__(name, backend)
        self.task = _get_torch_task(self.name)
        
    def get_theta_dim(self):
        return self.task.dim_parameters
    
    def get_x_dim(self):
        return self.task.dim_data

    def get_prior(self):
        if self.backend == "torch":
            return self.task.get_prior_dist()
        else:
            raise NotImplementedError()

    def get_simulator(self):
        if self.backend == "torch":
            return self.task.get_simulator()
        else:
            raise NotImplementedError()
    
    def get_node_id(self):
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch":
            return torch.arange(dim)
        else:
            return jnp.arange(dim)

    def get_data(self, num_samples: int, **kwargs):
        try:
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            return {"theta":thetas, "x":xs}
        except:
            # If not implemented in JAX, use PyTorch
            old_backed = self.backend
            self.backend = "torch"
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            self.backend = old_backed
            if self.backend == "numpy":
                thetas = thetas.numpy()
                xs = xs.numpy()
            elif self.backend == "jax":
                thetas = jnp.array(thetas)
                xs = jnp.array(xs)
            return {"theta":thetas, "x":xs}

    def get_observation(self, index: int):
        if self.backend == "torch":
            return self.task.get_observation(index)
        else:
            out = self.task.get_observation(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_reference_posterior_samples(self, index: int):
        if self.backend == "torch":
            return self.task.get_reference_posterior_samples(index)
        else:
            out = self.task.get_reference_posterior_samples(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_true_parameters(self, index: int):
        if self.backend == "torch":
            return self.task.get_true_parameters(index)
        else:
            out = self.task.get_true_parameters(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

class LinearGaussian(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_linear", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_i_mask = jnp.eye(x_dim, dtype=jnp.bool_)
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.eye((x_dim)), x_i_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
    
    
class BernoulliGLM(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="bernoulli_glm", backend=backend)
        
    def get_base_mask_fn(self):
        raise NotImplementedError()
    

class BernoulliGLMRaw(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="bernoulli_glm_raw", backend=backend)
        
    def get_base_mask_fn(self):
        raise NotImplementedError()

class MixtureGaussian(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="gaussian_mixture", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
        
    

class TwoMoons(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="two_moons", backend=backend)
        
    def get_base_mask_fn(self):
        task = self.task
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim, x_dim))], [jnp.ones((x_dim, theta_dim)), x_mask]])
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn
        

class SLCP(SBIBMTask):
    def __init__(self, backend: str = "torch") -> None:
        super().__init__(name="slcp", backend=backend)
        
    def get_base_mask_fn(self):
        task = _get_torch_task(self.name)
        theta_dim = task.dim_parameters
        x_dim = task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_) 
        # TODO This could be triangular -> DAG
        x_i_dim = x_dim // 4
        x_i_mask = jax.scipy.linalg.block_diag(*tuple([jnp.tril(jnp.ones((x_i_dim,x_i_dim), dtype=jnp.bool_))]*4)) 
        base_mask = jnp.block([[thetas_mask, jnp.zeros((theta_dim,x_dim))], [jnp.ones((x_dim, theta_dim)), x_i_mask]]) 
        base_mask = base_mask.astype(jnp.bool_)
        def base_mask_fn(node_ids, node_meta_data):
            # If node_ids are permuted, we need to permute the base_mask
            return base_mask[node_ids, :][:, node_ids]
        
        return base_mask_fn


# class AllConditionalBMTask(ABC):

#     def __init__(self, task_name: str) -> None:
#         self.task_name = task_name
#         self.task = sbibm.get_task(task_name)
#         self.base_mask_fn = self.get_base_mask_fn()
#         self.simulator = self.task.get_simulator()
#         self.prior = self.task.get_prior()
    
#     @abstractmethod
#     def get_base_mask_fn(self):
#         """
#         Returns a function that takes in node_ids and node_meta_data and returns the base mask
#         for the given node_ids.
#         """
#         pass

# class TwoMoonsAllConditionalTask(AllConditionalBMTask):
#     def __init__(self) -> None:
#         super().__init__(task_name="two_moons")
#         return

#     def get_base_mask_fn(self):
#         thetas_mask = jnp.eye(2, dtype=jnp.bool_)
#         x_mask = jnp.tril(jnp.ones((2, 2), dtype=jnp.bool_))
#         base_mask = jnp.block(
#             [[thetas_mask, jnp.zeros((2, 2))], [jnp.ones((2, 2)), x_mask]]
#         )
#         base_mask = base_mask.astype(jnp.bool_)

#         def base_mask_fn(node_ids, node_meta_data):
#             return base_mask[node_ids, :][:, node_ids]

#         return base_mask_fn


# class SLCPAllConditionalTask(AllConditionalBMTask):
#     def __init__(self) -> None:
#         super().__init__(task_name="slcp")
        
#         # def ravel_condition_mask(condition_mask):
#         #     thetas_cond, x1_cond, x2_cond, x3_cond, x4_cond = jnp.split(condition_mask, [5,7,9,11], axis=-1)
#         #     x1_cond = jnp.any(x1_cond, axis=-1)[None]
#         #     x2_cond = jnp.any(x2_cond, axis=-1)[None]
#         #     x3_cond = jnp.any(x3_cond, axis=-1)[None]
#         #     x4_cond = jnp.any(x4_cond, axis=-1)[None]
#         #     return jnp.hstack([thetas_cond, x1_cond, x2_cond, x3_cond, x4_cond])
#         # def unravel_condition_mask(condition_mask):
#         #     thetas_cond, x1_cond, x2_cond, x3_cond, x4_cond = jnp.split(condition_mask, [5,6,7,8], axis=-1)
#         #     x1_cond = jnp.repeat(x1_cond, 2, axis=-1)
#         #     x2_cond = jnp.repeat(x2_cond, 2, axis=-1)
#         #     x3_cond = jnp.repeat(x3_cond, 2, axis=-1)
#         #     x4_cond = jnp.repeat(x4_cond, 2, axis=-1)
#         #     return jnp.hstack([thetas_cond, x1_cond, x2_cond, x3_cond, x4_cond])
        
#         # self.ravel_condition_mask = ravel_condition_mask
#         # self.unravel_condition_mask = unravel_condition_mask
#         return

#     def get_base_mask_fn(self):
#         theta_dim = 5
#         x_dim = 8
#         thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
#         # TODO This could be triangular -> DAG
#         x_i_dim = x_dim // 4
#         x_i_mask = jax.scipy.linalg.block_diag(
#             *tuple([jnp.tril(jnp.ones((x_i_dim, x_i_dim), dtype=jnp.bool_))] * 4)
#         )
#         base_mask = jnp.block(
#             [
#                 [thetas_mask, jnp.zeros((theta_dim, x_dim))],
#                 [jnp.ones((x_dim, theta_dim)), x_i_mask],
#             ]
#         )
#         base_mask = base_mask.astype(jnp.bool_)

#         def base_mask_fn(node_ids, node_meta_data):
#             # If node_ids are permuted, we need to permute the base_mask
#             return base_mask[node_ids, :][:, node_ids]

#         return base_mask_fn


# class NonlinearGaussianTreeAllConditionalTask(AllConditionalBMTask):
#     def __init__(self) -> None:
#         super().__init__(task_name="two_moons")
#         return

#     def get_base_mask_fn(self):
#         base_mask = jnp.array(
#             [
#                 [True, False, False, False, False, False, False],
#                 [True, True, False, False, False, False, False],
#                 [True, False, True, False, False, False, False],
#                 [False, True, False, True, False, False, False],
#                 [False, True, False, False, True, False, False],
#                 [False, False, True, False, False, True, False],
#                 [False, False, True, False, False, False, True],
#             ]
#         )

#         def base_mask_fn(node_ids, node_meta_data):
#             return base_mask[node_ids, :][:, node_ids]

#         return base_mask_fn


# class NonlinearMarcovChainAllConditionalTask(AllConditionalBMTask):
#     def __init__(self) -> None:
#         super().__init__(task_name="two_moons")
#         return

#     def get_base_mask_fn(self):
#         # Marcovian structure
#         theta_mask = jnp.eye(10, dtype=jnp.bool_) | jnp.eye(10, k=-1, dtype=jnp.bool_)
#         xs_mask = jnp.eye(10, dtype=jnp.bool_)
#         theta_xs_mask = jnp.eye(10, dtype=jnp.bool_)
#         fill_mask = jnp.zeros((10, 10), dtype=jnp.bool_)
#         base_mask = jnp.block([[theta_mask, fill_mask], [theta_xs_mask, xs_mask]])

#         def base_mask_fn(node_ids, node_meta_data):
#             return base_mask[node_ids, :][:, node_ids]

#         return base_mask_fn