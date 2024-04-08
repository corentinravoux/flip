import numpy as np
import scipy as sc

from flip import vectors
from flip.utils import create_log

log = create_log()


def log_likelihood_gaussian_inverse(vector, covariance_sum):
    _, logdet = np.linalg.slogdet(covariance_sum)
    inverse_covariance_sum = np.linalg.inv(covariance_sum)
    chi2 = np.dot(vector, np.dot(inverse_covariance_sum, vector))
    return -0.5 * (vector.size * np.log(2 * np.pi) + logdet + chi2)


def log_likelihood_gaussian_cholesky(vector, covariance_sum):
    cholesky = sc.linalg.cho_factor(covariance_sum)
    logdet = 2 * np.sum(np.log(np.diag(cholesky[0])))
    chi2 = np.dot(vector, sc.linalg.cho_solve(cholesky, vector))
    return -0.5 * (vector.size * np.log(2 * np.pi) + logdet + chi2)


class BaseLikelihood(object):

    _default_likelihood_properties = {
        "inversion_method": "inverse",
        "velocity_type": "direct",
        "velocity_estimator": "full",
        "negative_log_likelihood": True,
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

        return likelihood

    def load_data_vector(
        self,
        model_type,
        parameter_values_dict,
    ):
        if model_type in ["velocity", "density_velocity", "full"]:
            velocity, velocity_error = vectors.load_velocity_vector(
                self.data,
                parameter_values_dict,
                velocity_type=self.likelihood_properties["velocity_type"],
                velocity_estimator=self.likelihood_properties["velocity_estimator"],
            )

        if model_type in ["density", "density_velocity", "full"]:
            density, density_error = vectors.load_density_vector(self.data)

        if model_type == "density":
            return density, density_error
        elif model_type == "velocity":
            return velocity, velocity_error
        elif model_type in ["density_velocity", "full"]:
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
        likelihood_properties={},
    ):
        super(MultivariateGaussianLikelihood, self).__init__(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
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
            parameter_values_dict, vector_error
        )
        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}"
        )
        if self.likelihood_properties["negative_log_likelihood"]:
            return -likelihood_function(vector, covariance_sum)
        return likelihood_function(vector, covariance_sum)


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

        super(MultivariateGaussianLikelihoodInterpolate1D, self).__init__(
            covariance=covariance,
            data=data,
            parameter_names=parameter_names,
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

    def __call__(self, parameter_values):
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
            self.covariance[0].model_type,
            parameter_values_dict,
        )

        covariance_sum_list = []
        for i in range(len(self.covariance)):
            covariance_sum_list.append(
                self.covariance[i].compute_covariance_sum(
                    parameter_values_dict, vector_error
                )
            )
        covariance_sum_interpolated = sc.interpolate.interp1d(
            self.interpolation_value_range, covariance_sum_list, copy=False, axis=0
        )
        covariance_sum = covariance_sum_interpolated(
            parameter_values_dict[self.interpolation_value_name]
        )

        likelihood_function = eval(
            f"log_likelihood_gaussian_{self.likelihood_properties['inversion_method']}"
        )
        if self.likelihood_properties["negative_log_likelihood"]:
            return -likelihood_function(vector, covariance_sum)
        return likelihood_function(vector, covariance_sum)


class MultivariateGaussianLikelihoodInterpolate2D(BaseLikelihood):
    def __init__(
        self,
        covariance=None,
        data=None,
        parameter_names=None,
        likelihood_properties={},
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
            likelihood_properties=likelihood_properties,
        )
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
        if self.likelihood_properties["negative_log_likelihood"]:
            return -likelihood_function(vector, covariance_sum)
        return likelihood_function(vector, covariance_sum)
