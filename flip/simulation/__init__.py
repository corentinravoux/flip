"""Init file of the flip.simulation package.

This package provides tools for forward-model cosmological inference using
differentiable N-body simulations.  The ``jaxpm``, ``jaxopt``, ``jax_cosmo``,
and ``diffrax`` packages are optional and can be installed with::

    pip install jaxpm jaxopt jax_cosmo diffrax
"""

from flip.utils import create_log

log = create_log()

try:
    from . import fitter, generate, likelihood
except ImportError as e:
    log.add(
        f"Could not import flip.simulation modules ({e}). "
        "Install the optional dependencies with: "
        "pip install jaxpm jaxopt jax_cosmo diffrax",
        level="warning",
    )
