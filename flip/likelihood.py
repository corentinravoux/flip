import numpy as np
from scipy.linalg import cho_factor, cho_solve
from flip.utils import create_log

from flip.covariance.adamsblake20 import coefficients as coefficients_adamsblake20
from flip.covariance.lai22 import coefficients as coefficients_lai22
from flip.covariance.carreres23 import coefficients as coefficients_carreres23
from flip.covariance.ravouxcarreres import coefficients as coefficients_ravouxcarreres

log = create_log()


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

    def compute_covariance_sum(
        self,
        parameter_values_dict,
    ):
        """
        The compute_covariance_sum function computes the sum of all covariance matrices
            and adds the diagonal terms.

        Args:
            self: Access the attributes of the class
            parameter_values_dict: Pass the values of the parameters
            : Compute the covariance matrix

        Returns:
            The sum of the covariance matrices with their respective coefficients

        """
        coefficients_dict = eval(
            f"coefficients_{self.covariance.model_name}.get_coefficients"
        )(
            self.covariance.model_type,
            parameter_values_dict,
        )
        coefficients_dict_diagonal = eval(
            f"coefficients_{self.covariance.model_name}.get_diagonal_coefficients"
        )(
            self.covariance.model_type,
            parameter_values_dict,
        )

        if self.covariance.model_type == "density":
            covariance_sum = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["gg"])
                ],
                axis=0,
            )
            covariance_sum += np.diag(
                coefficients_dict_diagonal["gg"] + self.vector_err**2
            )

        elif self.covariance.model_type == "velocity":
            covariance_sum = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["vv"])
                ],
                axis=0,
            )

            covariance_sum += np.diag(
                coefficients_dict_diagonal["vv"] + self.vector_err**2
            )

        elif self.covariance.model_type in ["density_velocity", "full"]:
            number_densities = self.covariance.number_densities
            number_velocities = self.covariance.number_velocities
            density_err = self.vector_err[:number_densities]
            velocity_err = self.vector_err[
                number_densities : number_densities + number_velocities
            ]

            if self.covariance.model_type == "density_velocity":
                covariance_sum_gv = np.zeros((number_densities, number_velocities))
            elif self.covariance.model_type == "full":
                covariance_sum_gv = np.sum(
                    [
                        coefficients_dict["gv"][i] * cov
                        for i, cov in enumerate(self.covariance.covariance_dict["gv"])
                    ],
                    axis=0,
                )
            covariance_sum_gg = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["gg"])
                ],
                axis=0,
            )
            covariance_sum_gg += np.diag(
                coefficients_dict_diagonal["gg"] + density_err**2
            )

            covariance_sum_vv = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["vv"])
                ],
                axis=0,
            )

            covariance_sum_vv += np.diag(
                coefficients_dict_diagonal["vv"] + velocity_err**2
            )

            covariance_sum = np.block(
                [
                    [covariance_sum_gg, covariance_sum_gv],
                    [covariance_sum_gv.T, covariance_sum_vv],
                ]
            )
        else:
            log.add(f"Wrong model type in the loaded covariance.")

        return covariance_sum

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

    def __call__(self, parameter_values):
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

        covariance_sum = self.compute_covariance_sum(
            parameter_values_dict,
        )
        cholesky = cho_factor(covariance_sum)
        logdet = 2 * np.sum(np.log(np.diag(cholesky[0])))
        chi2 = np.dot(self.vector, cho_solve(cholesky, self.vector))
        return 0.5 * (self.vector.size * np.log(2 * np.pi) + logdet + chi2)
