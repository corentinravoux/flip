import abc
from functools import partial

import numpy as np
import scipy as sc

from flip._config import __use_jax__
from flip.utils import create_log, prior_sum, return_prior

if __use_jax__:
    try:
        import jax.numpy as jnp
        import jax.scipy as jsc
        from jax import grad, jit

        jax_installed = True

    except ImportError:
        import numpy as jnp
        import scipy as jsc

        jax_installed = False
else:

    import numpy as jnp
    import scipy as jsc

    jax_installed = False


log = create_log()

_available_inversion_methods = [
    "inverse",
    "solve",
    "cholesky",
    "cholesky_regularized",
    "cholesky_inverse",
]


def log_likelihood_gaussian_inverse(vector, covariance_sum):
    r"""Compute multivariate Gaussian log-likelihood using explicit inverse.

    This evaluates $\mathcal{L} = -\tfrac{1}{2}[N\log(2\pi) + \log|C| + \chi^2]$
    with $\chi^2 = v^T C^{-1} v$ by explicitly inverting the covariance matrix.

    Args:
        vector (array-like): Residual data vector `v` of shape `(N,)`.
        covariance_sum (array-like): Total covariance matrix `C` of shape `(N, N)`.

    Returns:
        float: Log-likelihood value of the Gaussian model.

    Notes:
        Prefer solve or Cholesky variants for better numerical stability on ill-conditioned matrices.
    """
    _, logdet = jnp.linalg.slogdet(covariance_sum)
    inverse_covariance_sum = jnp.linalg.inv(covariance_sum)
    chi2 = jnp.dot(vector, jnp.dot(inverse_covariance_sum, vector))
    return -0.5 * (vector.size * jnp.log(2 * jnp.pi) + logdet + chi2)


def log_likelihood_gaussian_solve(vector, covariance_sum):
    r"""Compute multivariate Gaussian log-likelihood via linear solver.

    Uses `solve(C, v)` to avoid explicit inversion when computing $\chi^2 = v^T C^{-1} v$.

    Args:
        vector (array-like): Residual data vector `v` of shape `(N,)`.
        covariance_sum (array-like): Total covariance matrix `C` of shape `(N, N)`.

    Returns:
        float: Log-likelihood value of the Gaussian model.

    """
    _, logdet = jnp.linalg.slogdet(covariance_sum)
    chi2 = jnp.dot(vector.T, jnp.linalg.solve(covariance_sum, vector))
    return -0.5 * (vector.size * jnp.log(2 * jnp.pi) + logdet + chi2)


def log_likelihood_gaussian_cholesky(vector, covariance_sum):
    r"""Compute Gaussian log-likelihood using Cholesky factorization.

    Factorizes `C = L L^T` to compute both `log|C|` and $\chi^2$ stably.

    Args:
        vector (array-like): Residual data vector `v` of shape `(N,)`.
        covariance_sum (array-like): Positive-definite covariance matrix `C`.

    Returns:
        float: Log-likelihood value of the Gaussian model.

    """
    cholesky = jsc.linalg.cho_factor(covariance_sum)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(cholesky[0])))
    chi2 = jnp.dot(vector, jsc.linalg.cho_solve(cholesky, vector))
    return -0.5 * (vector.size * jnp.log(2 * jnp.pi) + logdet + chi2)


def log_likelihood_gaussian_cholesky_inverse(vector, covariance_sum):
    """Compute Gaussian log-likelihood using Cholesky, fallback to inverse.

    Attempts a Cholesky factorization; if it fails (non-PD matrix), falls back to
    explicit inversion-based computation.

    Args:
        vector (array-like): Residual data vector `v` of shape `(N,)`.
        covariance_sum (array-like): Covariance matrix `C` (ideally PD).

    Returns:
        float: Log-likelihood value of the Gaussian model.
    """
    try:
        return log_likelihood_gaussian_cholesky(vector, covariance_sum)
    except jnp.linalg.LinAlgError:
        return log_likelihood_gaussian_inverse(vector, covariance_sum)


def log_likelihood_gaussian_cholesky_regularized(vector, covariance_sum):
    """Compute Gaussian log-likelihood with eigenvalue regularization.

    Ensures positive-definiteness by replacing negative eigenvalues with their absolute
    values before Cholesky factorization.

    Args:
        vector (array-like): Residual data vector `v` of shape `(N,)`.
        covariance_sum (array-like): Covariance matrix `C` that may be indefinite.

    Returns:
        float: Log-likelihood value of the Gaussian model.
    """
    eigval, eigvec = jnp.linalg.eig(covariance_sum)
    cov_sum_regularized = eigvec @ jnp.abs(jnp.diag(eigval)) @ jnp.linalg.inv(eigvec)
    cholesky = jsc.linalg.cho_factor(cov_sum_regularized)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(cholesky[0])))
    chi2 = jnp.dot(vector, jsc.linalg.cho_solve(cholesky, vector))
    return -0.5 * (vector.size * jnp.log(2 * jnp.pi) + logdet + chi2)


if jax_installed:
    log_likelihood_gaussian_inverse_jit = jit(log_likelihood_gaussian_inverse)
    log_likelihood_gaussian_cholesky_jit = jit(log_likelihood_gaussian_cholesky)
    log_likelihood_gaussian_cholesky_inverse_jit = jit(
        log_likelihood_gaussian_cholesky_inverse
    )
    log_likelihood_gaussian_solve_jit = jit(log_likelihood_gaussian_solve)


class BaseLikelihood(abc.ABC):
    """Abstract base class for likelihood evaluation.

    Provides common setup for covariance verification, properties validation, prior
    initialization, and building the callable likelihood and optional gradient.

    Attributes:
        covariance (CovMatrix or list[CovMatrix]): Covariance model(s).
        data (object): Data provider with `free_par` and `give_data_and_variance`.
        parameter_names (list[str]): Ordered parameter names.
        free_par (list[str]): Combined free parameters from data and covariance.
        likelihood_properties (dict): Controls inversion method, sign, JIT, gradients.
        prior (Callable): Prior function returning log-prior given parameter dict.
        likelihood_call (Callable): Callable evaluating the likelihood.
        likelihood_grad (Callable|None): Gradient of likelihood if JAX is available.

    """

    _default_likelihood_properties = {
        "inversion_method": "inverse",
        "negative_log_likelihood": True,
        "use_jit": False,
        "use_gradient": False,
    }

    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        likelihood_properties={},
    ):
        self.covariance = covariance
        self.data = data
        self.parameter_names = parameter_names

        self.free_par = self.data.free_par[:]

        if isinstance(self.covariance, list):
            self.free_par += self.covariance[0].free_par
        else:
            self.free_par += self.covariance.free_par

        self.likelihood_properties = {
            **self._default_likelihood_properties,
            **likelihood_properties,
        }

        self.verify_covariance()
        self.verify_properties()
        self.prior = self.initialize_prior()

        self.likelihood_call, self.likelihood_grad = self._init_likelihood()

    def __call__(self, parameter_values):
        """Evaluate likelihood at parameter values.

        Args:
            parameter_values (array-like): Parameter vector aligned with `parameter_names`.

        Returns:
            float: Likelihood value, sign controlled by `negative_log_likelihood`.
        """
        return self.likelihood_call(parameter_values)

    @abc.abstractmethod
    def _init_likelihood(self, *args):
        """Initialize likelihood and optional gradient.

        Returns:
            tuple[Callable, Callable|None]: `(likelihood_call, likelihood_grad)`.
        """
        likelihood_fun = None
        likelihood_grad = None
        return likelihood_fun, likelihood_grad

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_names,
        likelihood_properties={},
        **kwargs,
    ):
        """Construct a likelihood instance from a covariance.

        Args:
            covariance (CovMatrix or list[CovMatrix]): Covariance model(s) to use.
            data (object): Data provider used by likelihood to build residuals/errors.
            parameter_names (list[str]): Parameter names ordering the input vector.
            likelihood_properties (dict, optional): Likelihood options overriding defaults.
            **kwargs: Extra arguments forwarded to subclass constructor.

        Returns:
            BaseLikelihood: Initialized likelihood instance.

        """

        likelihood = cls(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
            **kwargs,
        )

        return likelihood

    def initialize_prior(
        self,
    ):
        """Build prior function from likelihood properties.

        Returns:
            Callable: Prior function mapping parameter dict to log-prior.

        Raises:
            ValueError: If an unsupported prior type is requested.
        """
        if "prior" not in self.likelihood_properties.keys():
            return lambda x: 0
        else:
            prior_dict = self.likelihood_properties["prior"]
            priors = []
            for parameter_name, prior_properties in prior_dict.items():
                prior = return_prior(
                    parameter_name,
                    prior_properties,
                )
                priors.append(prior)

            prior_function = partial(prior_sum, priors)
            return prior_function

    def verify_covariance(self):
        """Ensure covariance matrices are ready for likelihood evaluation.

        Converts flat covariances to matrix form if required and initializes the
        cached `compute_covariance_sum` (and JIT variant) functions.
        """
        if isinstance(self.covariance, list):
            for i in range(len(self.covariance)):
                if self.covariance[i].matrix_form is False:
                    self.covariance[i].compute_matrix_covariance()
                if (self.covariance[i].compute_covariance_sum is None) or (
                    self.covariance[i].compute_covariance_sum_jit is None
                ):
                    self.covariance[i].init_compute_covariance_sum()
        else:
            if (
                self.covariance.matrix_form is False
                and self.covariance.emulator_flag is False
            ):
                self.covariance.compute_matrix_covariance()
            if (
                self.covariance.compute_covariance_sum is None
                or self.covariance.compute_covariance_sum_jit is None
            ):
                self.covariance.init_compute_covariance_sum()

    def verify_properties(self):
        """Validate likelihood properties such as inversion method.

        Raises:
            ValueError: If the inversion method is not supported.
        """
        if (
            self.likelihood_properties["inversion_method"]
            not in _available_inversion_methods
        ):
            raise ValueError(
                f"""The inversion method {self.likelihood_properties['inversion_method']} is not available. """
                f"""Please choose between {_available_inversion_methods}"""
            )


class MultivariateGaussianLikelihood(BaseLikelihood):
    """Gaussian likelihood for a single covariance model.

    Supports multiple inversion strategies and optional JAX JIT/grad for speed.
    """

    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        likelihood_properties={},
    ):
        super(MultivariateGaussianLikelihood, self).__init__(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )

    def _init_likelihood(self):
        """Build callable likelihood and optional gradient for Gaussian model.

        Returns:
            tuple[Callable, Callable|None]: `(likelihood_call, likelihood_grad)`.
        """

        use_jit = self.likelihood_properties["use_jit"]

        if jax_installed & use_jit:
            suffix = "_jit"
        else:
            suffix = ""

        give_data_and_variance = eval(f"self.data.give_data_and_variance{suffix}")
        compute_covariance_sum = eval(f"self.covariance.compute_covariance_sum{suffix}")

        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}{suffix}"
        )

        if jax_installed & use_jit:
            prior = jit(self.prior)
        else:
            prior = self.prior

        def likelihood_evaluation(
            parameter_values,
        ):
            """Evaluate likelihood for given parameters.

            Args:
                parameter_values (array-like): Parameter vector aligned to names.

            Returns:
                float: Likelihood value (sign depends on `negative_log_likelihood`).
            """
            parameter_values_dict = dict(zip(self.parameter_names, parameter_values))
            vector, vector_variance = give_data_and_variance(parameter_values_dict)
            covariance_sum = compute_covariance_sum(
                parameter_values_dict,
                vector_variance,
            )
            likelihood_value = likelihood_function(vector, covariance_sum) + prior(
                parameter_values_dict
            )

            if self.likelihood_properties["negative_log_likelihood"]:
                likelihood_value *= -1
            return likelihood_value

        if jax_installed:
            likelihood_grad = grad(likelihood_evaluation)
            if use_jit:
                likelihood_evaluation = jit(likelihood_evaluation)
                likelihood_grad = jit(likelihood_grad)
        else:
            likelihood_grad = None
        return likelihood_evaluation, likelihood_grad


class MultivariateGaussianLikelihoodInterpolate1D(BaseLikelihood):
    """Gaussian likelihood with 1D interpolation over precomputed covariances.

    Interpolates the covariance matrix across a scalar parameter grid to avoid
    regeneration during fits.
    """

    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        likelihood_properties={},
        interpolation_value_name=None,
        interpolation_value_range=None,
    ):
        """Initialize 1D interpolation likelihood.

        Args:
            covariance (list[CovMatrix]): Covariance models sampled along interpolation axis.
            data (object): Data provider with residuals and variance.
            parameter_names (list[str]): Parameter names ordering the input vector.
            likelihood_properties (dict, optional): Likelihood options overriding defaults.
            interpolation_value_name (str): Name of the interpolation parameter.
            interpolation_value_range (array-like): Sorted grid of interpolation values.

        Returns:
            None: Initializes attributes and base class.
        """
        self.interpolation_value_name = interpolation_value_name
        self.interpolation_value_range = interpolation_value_range

        super(MultivariateGaussianLikelihoodInterpolate1D, self).__init__(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )

        self.free_par = [interpolation_value_name] + self.free_par

    def _init_likelihood(self):
        """Build callable likelihood and optional gradient for 1D interpolation.

        Returns:
            tuple[Callable, Callable|None]: `(likelihood_call, likelihood_grad)`.
        """
        use_jit = self.likelihood_properties["use_jit"]

        if jax_installed & use_jit:
            suffix = "_jit"
        else:
            suffix = ""

        give_data_and_variance = eval(f"self.data.give_data_and_variance{suffix}")

        compute_covariance_sum_list = []
        for i in range(len(self.covariance)):
            compute_covariance_sum_list.append(
                eval(f"self.covariance[{i}].compute_covariance_sum{suffix}")
            )

        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}{suffix}"
        )

        interpolation_value_range = self.interpolation_value_range
        if jax_installed & use_jit:
            prior = jit(self.prior)
            interpolation_value_range = jnp.array(interpolation_value_range)
        else:
            prior = self.prior

        def likelihood_evaluation(
            parameter_values,
        ):
            """Evaluate likelihood with interpolated covariance.

            Performs linear interpolation between nearest covariances along the
            interpolation axis.

            Args:
                parameter_values (array-like): Parameter vector.

            Returns:
                float: Likelihood value including prior on interpolation range.
            """
            parameter_values_dict = dict(zip(self.parameter_names, parameter_values))
            interpolation_value = parameter_values_dict[self.interpolation_value_name]

            prior_interpolation_range = jsc.stats.uniform.logpdf(
                interpolation_value,
                loc=interpolation_value_range[0],
                scale=interpolation_value_range[-1] - interpolation_value_range[0],
            )

            vector, vector_variance = give_data_and_variance(parameter_values_dict)

            # INTERPOLATION
            upper_index = jnp.searchsorted(
                interpolation_value_range, interpolation_value
            )
            upper_index = jnp.min(
                jnp.array([upper_index, len(interpolation_value_range) - 1])
            )

            covariance_sum_list = jnp.array(
                [
                    compute_covariance_sum(
                        parameter_values_dict,
                        vector_variance,
                    )
                    for compute_covariance_sum in compute_covariance_sum_list
                ]
            )

            covariance_sum_upper = covariance_sum_list[upper_index]

            covariance_sum_lower = covariance_sum_list[upper_index - 1]

            fraction_interpolation = (
                interpolation_value_range[upper_index] - interpolation_value
            )
            fraction_interpolation /= (
                interpolation_value_range[upper_index]
                - interpolation_value_range[upper_index - 1]
            )

            covariance_sum = (
                1 - fraction_interpolation
            ) * covariance_sum_upper + fraction_interpolation * covariance_sum_lower
            # END INTERPOLATION

            likelihood_value = (
                likelihood_function(vector, covariance_sum)
                + prior(parameter_values_dict)
                + prior_interpolation_range
            )

            if self.likelihood_properties["negative_log_likelihood"]:
                likelihood_value *= -1
            return likelihood_value

        if jax_installed:
            likelihood_grad = grad(likelihood_evaluation)
            if use_jit:
                likelihood_evaluation = jit(likelihood_evaluation)
                likelihood_grad = jit(likelihood_grad)
        else:
            likelihood_grad = None

        return likelihood_evaluation, likelihood_grad
