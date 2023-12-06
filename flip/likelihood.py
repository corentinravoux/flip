import numpy as np
import scipy as sc

from flip import vectors
from flip.utils import create_log

log = create_log()


def log_likelihood_gaussian_inverse(vector, covariance_sum):
    _, logdet = np.linalg.slogdet(covariance_sum)
    inverse_covariance_sum = np.linalg.inv(covariance_sum)
    chi2 = np.dot(vector, np.dot(inverse_covariance_sum, vector))
    return 0.5 * (vector.size * np.log(2 * np.pi) + logdet + chi2)


def log_likelihood_gaussian_cholesky(vector, covariance_sum):
    cholesky = sc.linalg.cho_factor(covariance_sum)
    logdet = 2 * np.sum(np.log(np.diag(cholesky[0])))
    chi2 = np.dot(vector, sc.linalg.cho_solve(cholesky, vector))
    return 0.5 * (vector.size * np.log(2 * np.pi) + logdet + chi2)


class BaseLikelihood(object):
    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        likelihood_properties=None,
    ):
        self.covariance = covariance
        self.data = data
        self.parameter_names = parameter_names

        _default_likelihood_properties = {
            "inversion_method": "inverse",
            "velocity_type": "direct",
            "velocity_estimator": "full",
        }
        if likelihood_properties == None:
            likelihood_properties = _default_likelihood_properties
        else:
            for key in _default_likelihood_properties.keys():
                if key not in likelihood_properties.keys():
                    likelihood_properties[key] = _default_likelihood_properties[key]

        self.likelihood_properties = likelihood_properties

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        parameter_names,
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
        if covariance.full_matrix is False:
            covariance.compute_full_matrix()

        likelihood = cls(covariance=covariance, parameter_names=parameter_names)

        return likelihood

    def load_data_vector(
        self,
        parameter_values_dict,
    ):
        if self.covariance.model_type in ["velocity", "density_velocity", "full"]:
            velocity, velocity_error = vectors.load_velocity_vector(
                self.data,
                parameter_values_dict,
                velocity_type=self.likelihood_properties["velocity_type"],
                velocity_estimator=self.likelihood_properties["velocity_estimator"],
            )

        if self.covariance.model_type in ["density", "density_velocity", "full"]:
            density, density_error = vectors.load_density_vector(self.data)

        if self.covariance.model_type == "density":
            return density, density_error
        elif self.covariance.model_type == "velocity":
            return velocity, velocity_error
        elif self.covariance.model_type in ["density_velocity", "full"]:
            return (
                np.concatenate([density, velocity], axis=0),
                np.concatenate([density_error, velocity_error], axis=0),
            )
        else:
            log.add(f"Wrong model type in the loaded covariance.")


class MultivariateGaussianLikelihood(BaseLikelihood):
    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        likelihood_properties=None,
    ):
        super(MultivariateGaussianLikelihood, self).__init__(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )

    def __call__(self, parameter_values):
        parameter_values_dict = dict(zip(self.parameter_names, parameter_values))

        vector, vector_error = self.load_data_vector(
            parameter_values_dict,
        )
        covariance_sum = self.covariance.compute_covariance_sum(
            parameter_values_dict, vector_error
        )
        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}"
        )
        return likelihood_function(vector, covariance_sum)


class MultivariateGaussianLikelihoodInterpolate1D(BaseLikelihood):
    def __init__(
        self,
        covariance_list=None,
        data=None,
        parameter_names=None,
        likelihood_properties=None,
        interpolation_value_range=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.
        The __init__ function takes arguments (in this case, just self), but can take any number of additional arguments that are passed to it when a new instance of a class is created.

        Args:
            self: Represent the instance of the class
            covariance_list: Define the covariance matrix
            data: Store the data
            parameter_names: Define the parameters that are used in the likelihood
            likelihood_properties: Pass in the likelihood properties
            interpolation_value_range: Define the range of values that will be used to interpolate
            : Set the interpolation value range

        Returns:
            The object itself
        """
        super(MultivariateGaussianLikelihoodInterpolate1D, self).__init__(
            covariance_list=covariance_list,
            data=data,
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )
        self.interpolation_value_range = interpolation_value_range

    def __call__(self, parameter_values, interpolation_value):
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

        vector, vector_error = self.load_data_vector(
            parameter_values_dict,
        )

        covariance_sum_list = []
        for i in range(len(self.covariance_list)):
            covariance_sum_list.append(
                self.covariance_list[i].compute_covariance_sum(
                    parameter_values_dict, vector_error
                )
            )
        covariance_sum_interpolated = sc.interpolate.interp1d(
            self.interpolation_value_range, covariance_sum_list, copy=False, axis=0
        )
        covariance_sum = covariance_sum_interpolated(interpolation_value)

        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}"
        )
        return likelihood_function(vector, covariance_sum)


class MultivariateGaussianLikelihoodInterpolate2D(BaseLikelihood):
    def __init__(
        self,
        covariance_matrix=None,
        data=None,
        parameter_names=None,
        likelihood_properties=None,
        interpolation_value_range_0=None,
        interpolation_value_range_1=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.
        The __init__ function should always accept keyword arguments, as this allows for flexible instantiation of objects.

        Args:
            self: Represent the instance of the class
            covariance_matrix: Store the covariance matrix of the data
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
            covariance_matrix=covariance_matrix,
            data=data,
            parameter_names=parameter_names,
            likelihood_properties=likelihood_properties,
        )
        self.interpolation_value_range_0 = interpolation_value_range_0
        self.interpolation_value_range_1 = interpolation_value_range_1

    def __call__(
        self,
        parameter_values,
        interpolation_value_0,
        interpolation_value_1,
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

        vector, vector_error = self.load_data_vector(
            parameter_values_dict,
        )

        covariance_sum_matrix = []

        for i in range(len(self.covariance_matrix)):
            covariance_sum_matrix_i = []
            for j in range(len(self.covariance_matrix[i])):
                covariance_sum_matrix_i.append(
                    self.covariance_matrix[i][j].compute_covariance_sum(
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
        return likelihood_function(vector, covariance_sum)
