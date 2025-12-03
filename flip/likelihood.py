import abc
from functools import partial

import numpy as np
import scipy as sc

from flip.utils import create_log

from ._config import __use_jax__

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


_available_priors = ["gaussian", "positive", "uniform"]

_available_inversion_methods = [
    "inverse",
    "solve",
    "cholesky",
    "cholesky_regularized",
    "cholesky_inverse",
]


def log_likelihood_gaussian_inverse(vector, covariance_sum):
    _, logdet = jnp.linalg.slogdet(covariance_sum)
    inverse_covariance_sum = jnp.linalg.inv(covariance_sum)
    chi2 = jnp.dot(vector, jnp.dot(inverse_covariance_sum, vector))
    return -0.5 * (vector.size * jnp.log(2 * jnp.pi) + logdet + chi2)


def log_likelihood_gaussian_solve(vector, covariance_sum):
    _, logdet = jnp.linalg.slogdet(covariance_sum)
    chi2 = jnp.dot(vector.T, jnp.linalg.solve(covariance_sum, vector))
    return -0.5 * (vector.size * jnp.log(2 * jnp.pi) + logdet + chi2)


def log_likelihood_gaussian_cholesky(vector, covariance_sum):
    cholesky = jsc.linalg.cho_factor(covariance_sum)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(cholesky[0])))
    chi2 = jnp.dot(vector, jsc.linalg.cho_solve(cholesky, vector))
    return -0.5 * (vector.size * jnp.log(2 * jnp.pi) + logdet + chi2)


def log_likelihood_gaussian_cholesky_inverse(vector, covariance_sum):
    try:
        return log_likelihood_gaussian_cholesky(vector, covariance_sum)
    except jnp.linalg.LinAlgError:
        return log_likelihood_gaussian_inverse(vector, covariance_sum)


def log_likelihood_gaussian_cholesky_regularized(vector, covariance_sum):
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


def no_prior(x):
    return 0


def prior_sum(priors, x):
    return sum(prior(x) for prior in priors)


class BaseLikelihood(abc.ABC):

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
        return self.likelihood_call(parameter_values)

    @abc.abstractmethod
    def _init_likelihood(self, *args):
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
        """
        The init_from_covariance function is a class method that initializes the likelihood object from a covariance matrix.

        Args:
            cls: Create a new instance of the class
            covariance: Compute the full matrix of the covariance
            parameter_names: Set the names of the parameters
            density: Compute the vector and its error
            density_err: Compute the vector_err
            velocity: Compute the vector and vector_err
            velocity_err: Compute the error in the vector
            : Compute the vector

        Returns:
            A likelihood object

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
        if "prior" not in self.likelihood_properties.keys():
            return no_prior
        else:
            prior_dict = self.likelihood_properties["prior"]
            priors = []
            for parameter_name, prior_properties in prior_dict.items():
                if prior_properties["type"].lower() not in _available_priors:
                    raise ValueError(
                        f"""The prior type {prior_properties["type"]} is not available"""
                        f"""Please choose between {_available_priors}"""
                    )
                elif prior_properties["type"].lower() == "gaussian":
                    prior = GaussianPrior(
                        parameter_name=parameter_name,
                        prior_mean=prior_properties["mean"],
                        prior_standard_deviation=prior_properties["standard_deviation"],
                    )
                elif prior_properties["type"].lower() == "positive":
                    prior = PositivePrior(
                        parameter_name=parameter_name,
                    )
                elif prior_properties["type"].lower() == "uniform":
                    prior = UniformPrior(
                        parameter_name=parameter_name,
                        range=prior_properties["range"],
                    )
                priors.append(prior)

            prior_function = partial(prior_sum, priors)
            return prior_function

    def verify_covariance(self):
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
        if (
            self.likelihood_properties["inversion_method"]
            not in _available_inversion_methods
        ):
            raise ValueError(
                f"""The inversion method {self.likelihood_properties['inversion_method']} is not available. """
                f"""Please choose between {_available_inversion_methods}"""
            )


class MultivariateGaussianLikelihood(BaseLikelihood):
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
            covariance_prefactor_dict=None,
        ):
            parameter_values_dict = dict(zip(self.parameter_names, parameter_values))
            vector, vector_variance = give_data_and_variance(parameter_values_dict)
            covariance_sum = compute_covariance_sum(
                parameter_values_dict,
                vector_variance,
                covariance_prefactor_dict=covariance_prefactor_dict,
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
    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        likelihood_properties={},
        interpolation_value_name=None,
        interpolation_value_range=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.
        The __init__ function takes arguments, which are then assigned to object attributes:

        Args:
            self: Represent the instance of the class
            covariance: Set the covariance matrix of the likelihood
            data: Store the data
            parameter_names: Specify the names of the parameters that are used in this likelihood
            likelihood_properties: Pass in the interpolation_value_name and interpolation_value_range
            interpolation_value_name: Specify the name of the parameter that is being interpolated
            interpolation_value_range: Specify the range of values that will be used to interpolate
            : Define the interpolation value name

        Returns:
            The object itself
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
            covariance_prefactor_dict=None,
        ):
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

            covariance_sum_list = jnp.array(
                [
                    compute_covariance_sum(
                        parameter_values_dict,
                        vector_variance,
                        covariance_prefactor_dict=covariance_prefactor_dict,
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


# CR - This class is no deprecated, do not use it for now.


class MultivariateGaussianLikelihoodInterpolate2D(BaseLikelihood):
    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        prior=None,
        likelihood_properties={},
        interpolation_value_name_0=None,
        interpolation_value_name_1=None,
        interpolation_value_range_0=None,
        interpolation_value_range_1=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.
        The __init__ function should always accept keyword arguments, as this allows for flexible instantiation of objects.

        Args:
            self: Represent the instance of the class
            covariance: Store the covariance matrix of the data
            data: Set the data attribute of the class
            parameter_names: Specify the names of the parameters
            likelihood_properties: Pass in the interpolation values
            interpolation_value_range_0: Set the range of values that
            interpolation_value_range_1: Set the range of values that will be used to interpolate the likelihood
            : Set the interpolation value range for the first parameter

        Returns:
            The object itself
        """
        super(MultivariateGaussianLikelihoodInterpolate2D, self).__init__(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
            prior=prior,
            likelihood_properties=likelihood_properties,
        )
        self.interpolation_value_name_0 = interpolation_value_name_0
        self.interpolation_value_name_1 = interpolation_value_name_1
        self.interpolation_value_range_0 = interpolation_value_range_0
        self.interpolation_value_range_1 = interpolation_value_range_1

    def __call__(
        self,
        parameter_values,
        covariance_prefactor_dict=None,
    ):
        """
        The __call__ function is the function that will be called when the likelihood
        object is called. It takes in a list of parameter values, and returns a float
        value representing the log-likelihood value for those parameters. The __call__
        method should not be overwritten by subclasses unless you know what you are doing!

        Args:
            self: Refer to the object itself
            parameter_values: Compute the covariance matrix
            interpolation_value_0: Interpolate the covariance matrix along the first dimension
            interpolation_value_1: Interpolate the covariance matrix
            : Compute the covariance sum

        Returns:
            The log-likelihood function
        """
        parameter_values_dict = dict(zip(self.parameter_names, parameter_values))

        interpolation_value_0 = parameter_values_dict[self.interpolation_value_name_0]
        interpolation_value_1 = parameter_values_dict[self.interpolation_value_name_1]

        if (
            (interpolation_value_0 < self.interpolation_value_range_0[0])
            | (interpolation_value_0 > self.interpolation_value_range_0[-1])
            | (interpolation_value_1 < self.interpolation_value_range_1[0])
            | (interpolation_value_1 > self.interpolation_value_range_1[-1])
        ):
            if self.likelihood_properties["negative_log_likelihood"]:
                return np.inf
            else:
                return -np.inf

        vector, vector_variance = self.data(parameter_values)

        covariance_sum_matrix = []

        for i in range(len(self.covariance)):
            covariance_sum_matrix_i = []
            for j in range(len(self.covariance[i])):
                covariance_sum_matrix_i.append(
                    self.covariance[i][j].compute_covariance_sum(
                        parameter_values_dict,
                        vector_variance,
                        covariance_prefactor_dict=covariance_prefactor_dict,
                    )
                )
            covariance_sum_matrix.append(covariance_sum_matrix_i)

        value_00, value_11 = np.meshgrid(
            self.interpolation_value_range_0, self.interpolation_value_range_1
        )
        covariance_sum_interpolated = sc.interpolate.interp2d(
            value_00, value_11, covariance_sum_matrix, copy=False
        )
        covariance_sum = covariance_sum_interpolated(
            interpolation_value_0, interpolation_value_1
        )
        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}"
        )

        prior_value = self.prior(parameter_values_dict)
        if self.likelihood_properties["negative_log_likelihood"]:
            return -likelihood_function(vector, covariance_sum) - prior_value
        return likelihood_function(vector, covariance_sum) + prior_value


class Prior:
    def __init__(
        self,
        parameter_name=None,
    ):
        self.parameter_name = parameter_name


class GaussianPrior(Prior):

    def __init__(
        self,
        parameter_name=None,
        prior_mean=None,
        prior_standard_deviation=None,
    ):
        super().__init__(parameter_name=parameter_name)
        self.prior_mean = prior_mean
        self.prior_standard_deviation = prior_standard_deviation

    def __call__(
        self,
        parameter_values_dict,
    ):
        return -0.5 * (
            np.log(2 * jnp.pi * self.prior_standard_deviation**2)
            + (parameter_values_dict[self.parameter_name] - self.prior_mean) ** 2
            / self.prior_standard_deviation**2
        )


class PositivePrior(Prior):

    def __init__(
        self,
        parameter_name=None,
    ):
        super().__init__(parameter_name=parameter_name)

    def __call__(
        self,
        parameter_values_dict,
    ):
        return jnp.log(jnp.heaviside(parameter_values_dict[self.parameter_name], 0))


class UniformPrior(Prior):

    def __init__(self, parameter_name=None, range=None):
        super().__init__(parameter_name=parameter_name)
        self.range = range

    def __call__(
        self,
        parameter_values_dict,
    ):
        value = parameter_values_dict[self.parameter_name]
        return jsc.stats.uniform.logpdf(
            value, loc=self.range[0], scale=self.range[1] - self.range[0]
        )
