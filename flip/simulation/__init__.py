"""flip.simulation — JAX-differentiable forward modeling via JaxPM.

End-to-end pipeline for field-level inference of cosmological parameters
(in particular fσ₈) from peculiar velocity and density surveys.

Pipeline overview::

    cosmological params (omega_m, sigma8, …)
            │
            ▼
    get_cosmology() → jax_cosmo.Cosmology
            │
            ▼
    ForwardModel.get_fields(cosmo, seed)
      ├─ _differentiable_linear_field()   ← DC-mode-safe Gaussian IC
      ├─ _run_lpt()                        ← manual 1LPT (grad-safe)
      └─ [run_pm()]                        ← optional PM time-stepping
            │
     ┌──────┴──────┐
     ▼             ▼
 density_field  velocity_field  (on 3D mesh)
     │             │
     ▼             ▼
 interpolate_density_to_positions()
 interpolate_velocity_to_positions()
 compute_los_velocity()
     │
     ▼
 VelocityFieldLikelihood(cosmo_params) → -log L
     │
     ▼
 SimulationFitter.run() → best-fit params

Submodules
----------
generator
    :class:`ForwardModel` — DC-mode-safe IC + manual 1LPT + optional PM.
    :func:`get_cosmology` — build ``jax_cosmo.Cosmology``.
cosmo_utils
    Adapters between flip parameter dicts and ``jax_cosmo`` objects;
    P(k) callable construction from tabulated arrays.
painter
    CIC interpolation, LOS velocity projection, sky↔Cartesian conversion, RSD.
likelihood
    :class:`VelocityFieldLikelihood`, :func:`log_likelihood_gaussian`.
fitter
    :class:`SimulationFitter` — gradient-based optimization via ``jaxopt``.

Optional dependencies
---------------------
* ``jaxpm`` — particle mesh N-body (required for simulations).
* ``jax_cosmo`` — JAX cosmology (required for unit conversions).
* ``jaxopt`` — required for :class:`~flip.simulation.fitter.SimulationFitter`.
All are optional at import time; errors are raised at first use.
"""

# cosmo_utils has only numpy/scipy dependencies — always importable
from .cosmo_utils import (
    cosmo_with_sigma8_from_fs8,
    flip_params_to_jaxcosmo,
    make_pk_callable,
    make_pk_callable_from_dict,
    sigma8_from_fs8,
)

# painter only needs jax.numpy — fails gracefully without JAX
from .painter import (
    _cic_read,
    apply_rsd,
    cartesian_to_box_frame,
    compute_los_velocity,
    interpolate_density_to_positions,
    interpolate_velocity_to_positions,
    sky_to_cartesian,
)

# generator and likelihood/fitter require jaxpm + jax_cosmo — lazy-load
_LAZY = {
    "ForwardModel", "get_cosmology", "_differentiable_linear_field", "_run_lpt",
    "VelocityFieldLikelihood", "log_likelihood_gaussian",
    "SimulationFitter",
}


def __getattr__(name):
    if name in ("ForwardModel", "get_cosmology", "_differentiable_linear_field", "_run_lpt"):
        from .generator import ForwardModel, get_cosmology, _differentiable_linear_field, _run_lpt
        globals().update({
            "ForwardModel": ForwardModel, "get_cosmology": get_cosmology,
            "_differentiable_linear_field": _differentiable_linear_field,
            "_run_lpt": _run_lpt,
        })
        return globals()[name]
    if name in ("VelocityFieldLikelihood", "log_likelihood_gaussian"):
        from .likelihood import VelocityFieldLikelihood, log_likelihood_gaussian
        globals().update({
            "VelocityFieldLikelihood": VelocityFieldLikelihood,
            "log_likelihood_gaussian": log_likelihood_gaussian,
        })
        return globals()[name]
    if name == "SimulationFitter":
        from .fitter import SimulationFitter
        globals()["SimulationFitter"] = SimulationFitter
        return SimulationFitter
    raise AttributeError(f"module 'flip.simulation' has no attribute {name!r}")


__all__ = [
    # cosmo_utils
    "flip_params_to_jaxcosmo", "sigma8_from_fs8", "cosmo_with_sigma8_from_fs8",
    "make_pk_callable", "make_pk_callable_from_dict",
    # painter
    "_cic_read", "compute_los_velocity", "interpolate_velocity_to_positions",
    "interpolate_density_to_positions", "sky_to_cartesian",
    "cartesian_to_box_frame", "apply_rsd",
    # lazy
    "get_cosmology", "ForwardModel", "_differentiable_linear_field", "_run_lpt",
    "log_likelihood_gaussian", "VelocityFieldLikelihood",
    "SimulationFitter",
]
