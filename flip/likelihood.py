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
        super(MultivariateGaussianLikelihood, self).__init__(
            covariance=covariance,
            parameter_names=parameter_names,
            vector=vector,
            vector_err=vector_err,
        )

    def __call__(self, parameter_values):
        parameter_values_dict = dict(zip(self.parameter_names, parameter_values))

        covariance_sum = self.compute_covariance_sum(
            parameter_values_dict,
        )
        cholesky = cho_factor(covariance_sum)
        logdet = 2 * np.sum(np.log(np.diag(cholesky[0])))
        chi2 = np.dot(self.vector, cho_solve(cholesky, self.vector))
        return 0.5 * (self.vector.size * np.log(2 * np.pi) + logdet + chi2)
