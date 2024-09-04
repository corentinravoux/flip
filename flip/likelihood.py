from functools import partial

import numpy as np
import scipy as sc

try:
    import jax.numpy as jnp
    import jax.scipy as jsc
    from jax import jit

    jax_installed = True
except ImportError:
    import numpy as jnp
    import scipy as jsc

    jax_installed = False

from flip import vectors
from flip.utils import create_log

# try:
#     import jax, jaxlib
#     import jax.numpy as jnp
# except ImportError:
#     jax = None
#     import numpy as jnp


# def use_jax(array):
#     """Whether to use jax.numpy depending on whether array is jax's object"""
#     return jax and isinstance(array, (jaxlib.xla_extension.DeviceArrayBase, jax.core.Tracer))


# def np_jax(array):
#     """Return numpy or jax.numpy depending on whether array is jax's object"""
#     if use_jax(array):
#         return jnp
#     return np

# CR - cool implementation - to test

log = create_log()


_available_priors = ["gaussian", "positive", "uniform"]


def log_likelihood_gaussian_inverse(vector, covariance_sum):
    _, logdet = jnp.linalg.slogdet(covariance_sum)
    inverse_covariance_sum = jnp.linalg.inv(covariance_sum)
    chi2 = jnp.dot(vector, jnp.dot(inverse_covariance_sum, vector))
    return -0.5 * (vector.size * jnp.log(2 * np.pi) + logdet + chi2)


def log_likelihood_gaussian_cholesky(vector, covariance_sum):
    cholesky = jsc.linalg.cho_factor(covariance_sum)
    logdet = 2 * jnp.sum(jnp.log(jnp.diag(cholesky[0])))
    chi2 = jnp.dot(vector, jsc.linalg.cho_solve(cholesky, vector))
    return -0.5 * (vector.size * jnp.log(2 * np.pi) + logdet + chi2)


if jax_installed:
    log_likelihood_gaussian_inverse_jit = jit(log_likelihood_gaussian_inverse)
    log_likelihood_gaussian_cholesky_jit = jit(log_likelihood_gaussian_cholesky)


def no_prior(x):
    return 0


def prior_sum(priors, x):
    return sum(prior(x) for prior in priors)


def interpolate_covariance_sum_1d(
    interpolation_value_range,
    interpolation_value,
    covariance,
    parameter_values_dict,
    vector_error,
):
    if np.isnan(interpolation_value):
        return np.full_like(
            covariance[0].compute_covariance_sum(parameter_values_dict, vector_error),
            np.nan,
        )
    upper_index_interpolation = jnp.searchsorted(
        interpolation_value_range, interpolation_value
    )
    covariance_sum_upper = covariance[upper_index_interpolation].compute_covariance_sum(
        parameter_values_dict, vector_error
    )

    covariance_sum_lower = covariance[
        upper_index_interpolation - 1
    ].compute_covariance_sum(parameter_values_dict, vector_error)

    fraction_interpolation = (
        interpolation_value_range[upper_index_interpolation] - interpolation_value
    ) / (
        interpolation_value_range[upper_index_interpolation]
        - interpolation_value_range[upper_index_interpolation - 1]
    )
    covariance_sum = (
        1 - fraction_interpolation
    ) * covariance_sum_upper + fraction_interpolation * covariance_sum_lower
    return covariance_sum


class BaseLikelihood(object):

    _default_likelihood_properties = {
        "inversion_method": "inverse",
        "velocity_type": "direct",
        "velocity_estimator": "full",
        "negative_log_likelihood": True,
        "use_jit": False,
        "use_gradient": False,
    }

    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        prior=None,
        likelihood_properties={},
    ):
        self.covariance = covariance
        self.data = data
        self.parameter_names = parameter_names
        self.prior = prior

        self.likelihood_properties = {
            **self._default_likelihood_properties,
            **likelihood_properties,
        }

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

        likelihood.verify_covariance()

        likelihood.prior = likelihood.initialize_prior()

        return likelihood

    def load_data_vector(
        self,
        model_type,
        parameter_values_dict,
    ):
        if model_type in ["velocity", "density_velocity", "full"]:
            velocity, velocity_error = vectors.load_velocity_vectors(
                self.data,
                parameter_values_dict,
                velocity_type=self.likelihood_properties["velocity_type"],
                velocity_estimator=self.likelihood_properties["velocity_estimator"],
            )

        if model_type in ["density", "density_velocity", "full"]:
            density, density_error = vectors.load_density_vectors(self.data)

        if model_type == "density":
            vector, vector_error = density, density_error
        elif model_type == "velocity":
            vector, vector_error = velocity, velocity_error
        elif model_type in ["density_velocity", "full"]:
            vector = np.concatenate([density, velocity], axis=0)
            vector_error = np.concatenate([density_error, velocity_error], axis=0)
        else:
            log.add(f"Wrong model type in the loaded covariance.")
        return vector, vector_error

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


class MultivariateGaussianLikelihood(BaseLikelihood):
    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        prior=None,
        likelihood_properties={},
    ):
        super(MultivariateGaussianLikelihood, self).__init__(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
            prior=prior,
            likelihood_properties=likelihood_properties,
        )

    def verify_covariance(self):
        if self.covariance.full_matrix is False:
            self.covariance.compute_full_matrix()

    def __call__(self, parameter_values):
        parameter_values_dict = dict(zip(self.parameter_names, parameter_values))

        vector, vector_error = self.load_data_vector(
            self.covariance.model_type,
            parameter_values_dict,
        )
        covariance_sum = self.covariance.compute_covariance_sum(
            parameter_values_dict,
            vector_error,
            use_jit=self.likelihood_properties["use_jit"],
        )
        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}"
            + f"{'_jit' if jax_installed and self.likelihood_properties['use_jit'] else ''}"
        )
        prior_value = self.prior(parameter_values_dict)

        if self.likelihood_properties["negative_log_likelihood"]:
            likelihood_value = (
                -likelihood_function(vector, covariance_sum) - prior_value
            )
        else:
            likelihood_value = likelihood_function(vector, covariance_sum) + prior_value
        return likelihood_value


class MultivariateGaussianLikelihoodInterpolate1D(BaseLikelihood):
    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        prior=None,
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

        super(MultivariateGaussianLikelihoodInterpolate1D, self).__init__(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
            prior=prior,
            likelihood_properties=likelihood_properties,
        )
        self.interpolation_value_name = interpolation_value_name
        self.interpolation_value_range = interpolation_value_range

    def verify_covariance(self):
        """
        The verify_covariance function is used to ensure that the covariance matrix of each
            parameter in the model has been computed. If it has not, then this function will compute
            it and store it as a full matrix.

        Args:
            self: Bind the method to the object

        Returns:
            Nothing
        """
        for i in range(len(self.covariance)):
            if self.covariance[i].full_matrix is False:
                self.covariance[i].compute_full_matrix()

    def __call__(
        self,
        parameter_values,
    ):
        """
        The __call__ function is the function that is called when you call an instance of a class.
        For example, if you have a class named 'Foo' and create an instance of it like this:
            foo = Foo()
        then calling foo(x) will actually run the __call__ function in your Foo class with x as its argument.

        Args:
            self: Refer to the object itself
            parameter_values: Pass the values of the parameters to be used in this evaluation
            interpolation_value: Interpolate the covariance_sum

        Returns:
            The log likelihood value of the data vector given a set of parameters and an interpolation value
        """
        parameter_values_dict = dict(zip(self.parameter_names, parameter_values))

        interpolation_value = parameter_values_dict[self.interpolation_value_name]

        if (interpolation_value < self.interpolation_value_range[0]) | (
            interpolation_value > self.interpolation_value_range[-1]
        ):
            if self.likelihood_properties["negative_log_likelihood"]:
                return np.inf
            else:
                return -np.inf

        vector, vector_error = self.load_data_vector(
            self.covariance[0].model_type,
            parameter_values_dict,
        )

        covariance_sum = interpolate_covariance_sum_1d(
            self.interpolation_value_range,
            interpolation_value,
            self.covariance,
            parameter_values_dict,
            vector_error,
        )
        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}"
            + f"{'_jit' if jax_installed and self.likelihood_properties['use_jit'] else ''}"
        )
        prior_value = self.prior(parameter_values_dict)

        if self.likelihood_properties["negative_log_likelihood"]:
            likelihood_value = (
                -likelihood_function(vector, covariance_sum) - prior_value
            )
        else:
            likelihood_value = likelihood_function(vector, covariance_sum) + prior_value

        return likelihood_value


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
        super(MultivariateGaussianLikelihoodInterpolate1D, self).__init__(
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

    def verify_covariance(self):
        for i in range(len(self.covariance)):
            for j in range(len(self.covariance[i])):
                if self.covariance[i][j].full_matrix is False:
                    self.covariance[i][j].compute_full_matrix()

    def __call__(
        self,
        parameter_values,
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

        vector, vector_error = self.load_data_vector(
            self.covariance[0][0].model_type,
            parameter_values_dict,
        )

        covariance_sum_matrix = []

        for i in range(len(self.covariance)):
            covariance_sum_matrix_i = []
            for j in range(len(self.covariance[i])):
                covariance_sum_matrix_i.append(
                    self.covariance[i][j].compute_covariance_sum(
                        parameter_values_dict, vector_error
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
            np.log(2 * np.pi * self.prior_standard_deviation**2)
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
        if parameter_values_dict[self.parameter_name] < 0:
            return -np.inf
        else:
            return 0


class UniformPrior(Prior):

    def __init__(self, parameter_name=None, range=None):
        super().__init__(parameter_name=parameter_name)
        self.range = range

    def __call__(
        self,
        parameter_values_dict,
    ):
        value = parameter_values_dict[self.parameter_name]
        if (value < self.range[0]) | (value > self.range[1]):
            return -np.inf
        else:
            return 0
