import numpy as np
import scipy as sc

from flip.utils import create_log

log = create_log()


def log_likelihood_gaussian(
    vector,
    covariance_sum,
    inversion_method="inverse",
):
    if inversion_method == "inverse":
        return log_likelihood_gaussian_inverse(vector, covariance_sum)
    elif inversion_method == "cholesky":
        return log_likelihood_gaussian_cholesky(vector, covariance_sum)


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
        parameter_names=None,
        vector=None,
        vector_err=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the object with all of its properties and methods.

        Args:
            self: Represent the instance of the class
            covariance: Define the covariance matrix
            parameter_names: Set the names of the parameters
            vector: Set the vector of parameters
            vector_err: Set the error on each parameter
            : Define the covariance matrix

        Returns:
            None

        """
        self.covariance = covariance
        self.parameter_names = parameter_names
        self.vector = vector
        self.vector_err = vector_err

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        parameter_names,
        density=None,
        density_err=None,
        velocity=None,
        velocity_err=None,
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

        likelihood.vector = likelihood.compute_vector(
            density=density,
            velocity=velocity,
        )
        likelihood.vector_err = likelihood.compute_vector_err(
            density_err=density_err,
            velocity_err=velocity_err,
        )
        return likelihood

    def compute_vector(
        self,
        density=None,
        velocity=None,
    ):
        """
        The compute_vector function is used to create a vector of the density and velocity
            for each cell in the domain. This vector is then used to compute the covariance matrix
            using a model type that can be either &quot;density&quot;, &quot;velocity&quot;, or &quot;full&quot;. The full model
            type uses both density and velocity, while the other two only use one of them.

        Args:
            self: Access the attributes of the class
            density: Compute the density vector
            velocity: Compute the velocity vector
            : Define the type of covariance model

        Returns:
            The density and velocity vectors

        """
        if self.covariance.model_type == "density":
            return density
        elif self.covariance.model_type == "velocity":
            return velocity
        elif self.covariance.model_type in ["density_velocity", "full"]:
            return np.concatenate([density, velocity], axis=0)
        else:
            log.add(f"Wrong model type in the loaded covariance.")

    def compute_vector_err(
        self,
        density_err=None,
        velocity_err=None,
    ):
        """
        The compute_vector_err function is used to compute the error vector for a given model type.

        Args:
            self: Bind the method to an object
            density_err: Compute the density error
            velocity_err: Compute the velocity error
            : Create a new instance of the class

        Returns:
            The error in the density or velocity

        """
        if self.covariance.model_type == "density":
            return density_err
        elif self.covariance.model_type == "velocity":
            return velocity_err
        elif self.covariance.model_type in ["density_velocity", "full"]:
            return np.concatenate([density_err, velocity_err], axis=0)
        else:
            log.add(f"Wrong model type in the loaded covariance.")


class MultivariateGaussianLikelihood(BaseLikelihood):
    def __init__(
        self,
        covariance=None,
        parameter_names=None,
        vector=None,
        vector_err=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.
        The __init__ function should always accept at least one argument, self, which refers to the instance of the object being created.

        Args:
            self: Represent the instance of the class
            covariance: Specify the covariance matrix
            parameter_names: Set the names of the parameters
            vector: Set the mean of the multivariate gaussian
            vector_err: Set the error on each of the parameters in vector
            : Specify the covariance matrix of the data

        Returns:
            Nothing, but it does set the attributes of the object

        """
        super(MultivariateGaussianLikelihood, self).__init__(
            covariance=covariance,
            parameter_names=parameter_names,
            vector=vector,
            vector_err=vector_err,
        )

    def __call__(self, parameter_values, inversion_method="inverse"):
        """
        The __call__ function is the function that will be called when you call an instance of a class.

        Args:
            self: Make the function a method of the class
            parameter_values: Compute the covariance sum
            inversion_method: Specify the method used to invert the covariance matrix

        Returns:
            The log likelihood of the gaussian
        """

        parameter_values_dict = dict(zip(self.parameter_names, parameter_values))
        covariance_sum = self.covariance.compute_covariance_sum(
            parameter_values_dict, self.vector_err
        )
        return log_likelihood_gaussian(
            self.vector, covariance_sum, inversion_method=inversion_method
        )


class MultivariateGaussianLikelihoodInterpolate1D(BaseLikelihood):
    def __init__(
        self,
        covariance_list=None,
        parameter_names=None,
        vector=None,
        vector_err=None,
        interpolation_value_range=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.
        The __init__ function should always accept at least one argument, self, which refers to the instance of the object being created.

        Args:
            self: Represent the instance of the class
            covariance_list: Store the covariance matrix
            parameter_names: Set the names of the parameters
            vector: Store the mean of the multivariate gaussian
            vector_err: Set the error on each of the data points
            interpolation_value_range: Set the value of the parameter
            : Define the interpolation value

        Returns:
            An instance of the class
        """

        super(MultivariateGaussianLikelihoodInterpolate1D, self).__init__(
            covariance_list=covariance_list,
            parameter_names=parameter_names,
            vector=vector,
            vector_err=vector_err,
            interpolation_value_range=interpolation_value_range,
        )

    def __call__(
        self, parameter_values, interpolation_value, inversion_method="inverse"
    ):
        """
        The __call__ function is the function that will be called when you call
        the class instance. It takes a list of parameter values as input and returns
        the log likelihood value for those parameters. The __call__ function should
        be written in such a way that it can take any number of parameters, but we'll
        only ever pass it the number of parameters specified by self.parameter_names.

        Args:
            self: Access the attributes of the class
            parameter_values: Create a dictionary of parameter names and values

        Returns:
            The log-likelihood of the data given a set of parameter values

        """
        parameter_values_dict = dict(zip(self.parameter_names, parameter_values))

        covariance_sum_list = []
        for i in range(len(self.covariance_list)):
            covariance_sum_list.append(
                self.covariance_list[i].compute_covariance_sum(
                    parameter_values_dict, self.vector_err
                )
            )
        covariance_sum_interpolated = sc.interpolate.interp1d(
            self.interpolation_value_range, covariance_sum_list, copy=False, axis=0
        )
        covariance_sum = covariance_sum_interpolated(interpolation_value)

        return log_likelihood_gaussian(
            self.vector, covariance_sum, inversion_method=inversion_method
        )


class MultivariateGaussianLikelihoodInterpolate2D(BaseLikelihood):
    def __init__(
        self,
        covariance_matrix=None,
        parameter_names=None,
        vector=None,
        vector_err=None,
        interpolation_value_range_0=None,
        interpolation_value_range_1=None,
    ):
        """
        The __init__ function is called when the class is instantiated.
        It sets up the instance of the class, and defines all its attributes.
        The __init__ function should always accept at least one argument, self, which refers to the instance of the object being created.

        Args:
            self: Represent the instance of the class
            covariance_matrix: Store the covariance matrix
            parameter_names: Set the names of the parameters
            vector: Store the mean of the multivariate gaussian
            vector_err: Set the error on each of the data points
            interpolation_value_range: Set the value of the parameter
            : Define the interpolation value

        Returns:
            An instance of the class
        """

        super(MultivariateGaussianLikelihoodInterpolate1D, self).__init__(
            covariance_matrix=covariance_matrix,
            parameter_names=parameter_names,
            vector=vector,
            vector_err=vector_err,
            interpolation_value_range_0=interpolation_value_range_0,
            interpolation_value_range_1=interpolation_value_range_1,
        )

    def __call__(
        self,
        parameter_values,
        interpolation_value_0,
        interpolation_value_1,
        inversion_method="inverse",
    ):
        """
        The __call__ function is the function that will be called when you call
        the class instance. It takes a list of parameter values as input and returns
        the log likelihood value for those parameters. The __call__ function should
        be written in such a way that it can take any number of parameters, but we'll
        only ever pass it the number of parameters specified by self.parameter_names.

        Args:
            self: Access the attributes of the class
            parameter_values: Create a dictionary of parameter names and values

        Returns:
            The log-likelihood of the data given a set of parameter values

        """
        parameter_values_dict = dict(zip(self.parameter_names, parameter_values))

        covariance_sum_matrix = []

        for i in range(len(self.covariance_matrix)):
            covariance_sum_matrix_i = []
            for j in range(len(self.covariance_matrix[i])):
                covariance_sum_matrix_i.append(
                    self.covariance_matrix[i][j].compute_covariance_sum(
                        parameter_values_dict, self.vector_err
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

        return log_likelihood_gaussian(
            self.vector, covariance_sum, inversion_method=inversion_method
        )
