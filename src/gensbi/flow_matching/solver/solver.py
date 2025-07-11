from abc import ABC, abstractmethod
from jax import Array


class Solver(ABC):
    """Abstract base class for solvers."""

    @abstractmethod
    def sample(self, x_0: Array) -> Array:
        ...
