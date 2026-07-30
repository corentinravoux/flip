"""Global configuration flags for flip.

``__use_jax__`` selects the array backend: when True (and JAX is importable) the
JAX-accelerated code paths and their JIT variants are used, otherwise flip falls
back to NumPy/SciPy. Set ``flip._config.__use_jax__ = False`` before importing
subpackages to force the NumPy backend.
"""

__use_jax__ = True
