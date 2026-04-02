"""Gradient-based optimizer for the simulation likelihood.

:class:`SimulationFitter` minimizes the negative log-likelihood of a
:class:`~flip.simulation.likelihood.VelocityFieldLikelihood` (or any callable
returning a scalar loss) using gradient-based solvers from ``jaxopt``.

All solvers use ``jax.grad`` through the forward model, so full end-to-end
automatic differentiation is available.

Example::

    from flip.simulation.fitter import SimulationFitter

    fitter = SimulationFitter(
        likelihood=lik,
        initial_params={"omega_m": 0.3, "sigma8": 0.8},
        bounds=({"omega_m": 0.1, "sigma8": 0.3},
                {"omega_m": 0.9, "sigma8": 1.5}),
        solver="LBFGSB",
        maxiter=200,
    )
    best_params = fitter.run()
"""

import jax.numpy as jnp
import jaxopt

from flip.utils import create_log

log = create_log()

_AVAILABLE_SOLVERS = {
    "LBFGS": jaxopt.LBFGS,
    "LBFGSB": jaxopt.LBFGSB,
    "BFGS": jaxopt.BFGS,
    "GradientDescent": jaxopt.GradientDescent,
}


class SimulationFitter:
    """Minimize the simulation likelihood over cosmological parameters.

    Parameters are represented as a flat JAX array internally and converted
    to/from a dict for the likelihood interface.

    Args:
        likelihood (callable): Callable ``(params_dict) → scalar loss``.
            Typically a :class:`~flip.simulation.likelihood.VelocityFieldLikelihood`.
        initial_params (dict): Initial cosmological parameter values,
            e.g. ``{"omega_m": 0.3, "sigma8": 0.8}``.
        bounds (tuple[dict, dict] | None): Optional box constraints
            ``(lower_dict, upper_dict)`` with the same keys as
            ``initial_params``. Only used when ``solver="LBFGSB"``.
        solver (str): jaxopt solver name: ``"LBFGS"`` (default), ``"LBFGSB"``,
            ``"BFGS"``, or ``"GradientDescent"``.
        maxiter (int): Maximum optimizer iterations. Default 100.
        solver_kwargs (dict | None): Extra kwargs forwarded to the jaxopt
            solver constructor (e.g. ``tol``, ``stepsize``).

    Raises:
        ValueError: If ``solver`` is not one of the supported names.
    """

    def __init__(
        self,
        likelihood,
        initial_params,
        bounds=None,
        solver="LBFGS",
        maxiter=100,
        solver_kwargs=None,
    ):
        if solver not in _AVAILABLE_SOLVERS:
            raise ValueError(
                f"Solver '{solver}' not supported. "
                f"Choose from: {list(_AVAILABLE_SOLVERS.keys())}"
            )
        self.likelihood = likelihood
        self.initial_params = initial_params
        self.bounds = bounds
        self.solver_name = solver
        self.maxiter = maxiter
        self.solver_kwargs = solver_kwargs or {}
        self._result = None
        self._param_names = list(initial_params.keys())

    def _to_array(self, params_dict):
        return jnp.array([params_dict[k] for k in self._param_names])

    def _to_dict(self, params_array):
        return {k: float(params_array[i]) for i, k in enumerate(self._param_names)}

    def _objective(self, params_array):
        return self.likelihood(self._to_dict(params_array))

    def run(self):
        """Run the optimization and return the best-fit parameter dict.

        The raw jaxopt result is stored in :attr:`result` after completion.

        Returns:
            dict: Best-fit parameters with the same keys as ``initial_params``.
        """
        initial_array = self._to_array(self.initial_params)

        solver_cls = _AVAILABLE_SOLVERS[self.solver_name]
        kwargs = {"fun": self._objective, "maxiter": self.maxiter, **self.solver_kwargs}
        solver = solver_cls(**kwargs)

        if self.bounds is not None and self.solver_name == "LBFGSB":
            lower = jnp.array([self.bounds[0].get(k, -jnp.inf) for k in self._param_names])
            upper = jnp.array([self.bounds[1].get(k, jnp.inf) for k in self._param_names])
            result = solver.run(initial_array, bounds=(lower, upper))
        else:
            result = solver.run(initial_array)

        self._result = result

        try:
            log.info(
                "SimulationFitter (%s): %d iterations, final loss=%.6g",
                self.solver_name,
                result.state.iter_num,
                result.state.value,
            )
        except AttributeError:
            log.info("SimulationFitter (%s): optimization complete.", self.solver_name)

        return self._to_dict(result.params)

    @property
    def result(self):
        """Raw jaxopt result from the last :meth:`run` call, or ``None``."""
        return self._result
