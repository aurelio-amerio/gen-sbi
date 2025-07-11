"""
Simformer Model in GenSBI
-------------------------

This package provides the Simformer transformer-based model and related loss functions for simulation-based inference. The architecture is derived from the following foundational work:

* M. Gloeckler et al. "All-in-one simulation-based inference." `arXiv:2404.09636 <https://arxiv.org/abs/2404.09636>`_
* `mackelab/simformer <https://github.com/mackelab/simformer>`_
"""

# This file is a derivative work based on the Simformer architecture from the "All-in-one simulation-based inference" paper.
# Substantial modifications and extensions by Aurelio Amerio, 2025.
# If you use this package, please consider citing the original Simformer paper.


from .simformer import Simformer, SimformerParams, SimformerConditioner
from .loss import SimformerCFMLoss

__all__ = [
    "Simformer",
    "SimformerParams",
    "SimformerConditioner",
    "SimformerCFMLoss",
]