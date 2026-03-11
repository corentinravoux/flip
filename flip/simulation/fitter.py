"""Minimization of the simulation likelihood using jaxopt.

This module provides a :class:`SimulationFitter` that minimizes the negative
log-likelihood of a JaxPM forward simulation given observed peculiar velocity
data using gradient-based optimizers from the ``jaxopt`` library.

All optimizers support automatic differentiation through the JAX computational
graph, enabling gradient-based optimization over cosmological parameters.

Examples:
    >>> from flip.simulation.fitter import SimulationFitter
    >>> fitter = SimulationFitter(
    ...     likelihood=lik,
    ...     initial_params={"omega_m": 0.3, "sigma8": 0.8},
    ...     solver="LBFGS",
    ...     maxiter=200,
    ... )
    >>> best_params = fitter.run()
    >>> print(best_params)
"""

import jax.numpy as jnp
import jaxopt

from flip.utils import create_log

log = create_log()

#: Mapping from solver name string to the corresponding jaxopt class.
_AVAILABLE_SOLVERS = {
    "LBFGS": jaxopt.LBFGS,
    "LBFGSB": jaxopt.LBFGSB,
    "BFGS": jaxopt.BFGS,
    "GradientDescent": jaxopt.GradientDescent,
}


class SimulationFitter:
    """Minimize the simulation likelihood over cosmological parameters.

    Uses gradient-based optimization from ``jaxopt`` to find the cosmological
    parameter values that maximize the likelihood of the observed velocity
    field under the JaxPM forward simulation.

    Parameters are represented internally as a flat JAX array during
    optimization and converted to/from a parameter dictionary for the
    likelihood interface.

    Args:
        likelihood (callable): Callable that accepts a parameter dict and
            returns a scalar loss value (negative log-likelihood).  Typically
            an instance of
            :class:`~flip.simulation.likelihood.VelocityFieldLikelihood`.
        initial_params (dict): Initial cosmological parameter values, e.g.
            ``{"omega_m": 0.3, "sigma8": 0.8}``.  All values must be
            Python/NumPy floats (not JAX arrays) at construction time.
        bounds (tuple[dict, dict] | None): Optional box constraints as
            ``(lower_bounds_dict, upper_bounds_dict)`` where each dict has
            the same keys as ``initial_params``.  Only applied when
            ``solver="LBFGSB"``.  Default ``None`` (unconstrained).
        solver (str): Name of the jaxopt solver to use.  One of ``"LBFGS"``
            (default), ``"LBFGSB"``, ``"BFGS"``, or
            ``"GradientDescent"``.
        maxiter (int): Maximum number of optimizer iterations. Default 100.
        solver_kwargs (dict | None): Additional keyword arguments forwarded
            to the jaxopt solver constructor (e.g. ``tol``, ``stepsize``).

    Raises:
        ValueError: If ``solver`` is not one of the supported names.

    Examples:
        >>> fitter = SimulationFitter(
        ...     likelihood=lik,
        ...     initial_params={"omega_m": 0.3, "sigma8": 0.8},
        ...     bounds=({"omega_m": 0.1, "sigma8": 0.3},
        ...             {"omega_m": 0.9, "sigma8": 1.5}),
        ...     solver="LBFGSB",
        ...     maxiter=200,
        ... )
        >>> best_params = fitter.run()
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
                f"Solver '{solver}' is not supported. "
                f"Choose one of: {list(_AVAILABLE_SOLVERS.keys())}"
            )
        self.likelihood = likelihood
        self.initial_params = initial_params
        self.bounds = bounds
        self.solver_name = solver
        self.maxiter = maxiter
        self.solver_kwargs = solver_kwargs or {}
        self._result = None

        # Keep an ordered list of parameter names for array conversion
        self._param_names = list(initial_params.keys())

    # ------------------------------------------------------------------
    # Internal helpers for dict <-> array conversion
    # ------------------------------------------------------------------

    def _to_array(self, params_dict):
        """Convert parameter dict to a flat JAX array.

        Args:
            params_dict (dict): Parameter name -> value mapping.

        Returns:
            jnp.ndarray: 1-D array of shape ``(n_params,)``.
        """
        return jnp.array([params_dict[k] for k in self._param_names])

    def _to_dict(self, params_array):
        """Convert a flat array back to a parameter dict.

        Args:
            params_array (jnp.ndarray): 1-D array of shape ``(n_params,)``.

        Returns:
            dict: Parameter name -> scalar value mapping.
        """
        return {k: params_array[i] for i, k in enumerate(self._param_names)}

    def _objective(self, params_array):
        """Wrap the likelihood so it accepts a flat array.

        Args:
            params_array (jnp.ndarray): 1-D parameter array.

        Returns:
            float: Negative log-likelihood.
        """
        return self.likelihood(self._to_dict(params_array))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self):
        """Run the optimization and return the best-fit parameters.

        After completion, the raw jaxopt result is available via
        :attr:`result`.

        Returns:
            dict: Best-fit cosmological parameter values, with the same keys
            as ``initial_params``.
        """
        initial_array = self._to_array(self.initial_params)

        solver_cls = _AVAILABLE_SOLVERS[self.solver_name]
        kwargs = {
            "fun": self._objective,
            "maxiter": self.maxiter,
            **self.solver_kwargs,
        }

        solver = solver_cls(**kwargs)

        if self.bounds is not None and self.solver_name == "LBFGSB":
            lower = jnp.array(
                [self.bounds[0].get(k, -jnp.inf) for k in self._param_names]
            )
            upper = jnp.array(
                [self.bounds[1].get(k, jnp.inf) for k in self._param_names]
            )
            result = solver.run(initial_array, bounds=(lower, upper))
        else:
            result = solver.run(initial_array)
        self._result = result

        try:
            log.add(
                f"SimulationFitter ({self.solver_name}) finished after "
                f"{result.state.iter_num} iterations. "
                f"Final loss: {result.state.value:.6g}"
            )
        except AttributeError:
            log.add(f"SimulationFitter ({self.solver_name}) optimization complete.")

        return self._to_dict(result.params)

    @property
    def result(self):
        """Raw jaxopt result from the last call to :meth:`run`.

        Returns:
            jaxopt.OptStep | None: Result object, or ``None`` if :meth:`run`
            has not been called yet.
        """
        return self._result
