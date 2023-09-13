import numpy as np
from scipy.linalg import cho_factor, cho_solve
from flip.utils import create_log

log = create_log()


def get_coefficients_adamsblake20(parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["gg"] = [
        parameter_values_dict["bs8"] ** 2,
        parameter_values_dict["bs8"] * parameter_values_dict["fs8"],
        parameter_values_dict["fs8"] ** 2,
    ]
    coefficients_dict["gv"] = [
        parameter_values_dict["bs8"] * parameter_values_dict["fs8"],
        parameter_values_dict["fs8"] ** 2,
    ]
    coefficients_dict["vv"] = [parameter_values_dict["fs8"] ** 2]
    return coefficients_dict


def get_diagonal_coefficients_adamsblake20(parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    coefficients_dict["gg"] = 1.0
    return coefficients_dict


class BaseLikelihood(object):
    def __init__(
        self,
        likelihood=None,
        covariance=None,
    ):
        self.likelihood = likelihood
        self.covariance = covariance

    def compute_covariance_sum(
        self,
        parameter_values_dict,
        density_err=None,
        velocity_err=None,
    ):
        coefficients_dict = eval(f"get_coefficients_{self.covariance.model_name}")(
            parameter_values_dict
        )
        coefficients_dict_diagonal = eval(
            f"get_diagonal_coefficients_{self.covariance.model_name}"
        )(parameter_values_dict)

        if self.covariance.model_type == "density":
            covariance_sum = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["gg"])
                ],
                axis=0,
            )
            covariance_sum += np.diag(
                coefficients_dict_diagonal["gg"] * density_err**2
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
                coefficients_dict_diagonal["vv"] * velocity_err**2
            )

        elif self.covariance.model_type in ["density_velocity", "full"]:
            if self.covariance.model_type == "density_velocity":
                covariance_sum_gv = np.zeros(
                    (
                        self.covariance.covariance_dict["gg"][0].shape[0],
                        self.covariance.covariance_dict["vv"][0].shape[0],
                    )
                )
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
                coefficients_dict_diagonal["gg"] * density_err**2
            )
            covariance_sum_vv = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["vv"])
                ],
                axis=0,
            )
            covariance_sum_vv += np.diag(
                coefficients_dict_diagonal["vv"] * velocity_err**2
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
        likelihood=None,
        covariance=None,
    ):
        super(MultivariateGaussianLikelihood, self).__init__(
            likelihood=likelihood,
            covariance=covariance,
        )

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        density=None,
        density_err=None,
        velocity=None,
        velocity_err=None,
    ):
        multivariate_gaussian_likelihood = cls(covariance=covariance)
        vector = multivariate_gaussian_likelihood.compute_vector(
            density=density,
            velocity=velocity,
        )

        def log_likelihood(parameter_values_dict):
            covariance_sum = multivariate_gaussian_likelihood.compute_covariance_sum(
                parameter_values_dict,
                density_err=density_err,
                velocity_err=velocity_err,
            )
            cholesky = cho_factor(covariance_sum)
            logdet = 2 * np.sum(np.log(np.diag(cholesky[0])))
            chi2 = np.dot(vector, cho_solve(cholesky, vector))
            return -0.5 * (vector * np.log(2 * np.pi) + logdet + chi2)

        multivariate_gaussian_likelihood.likelihood = log_likelihood
        return multivariate_gaussian_likelihood


# CR - cleaner with factory etc:

# class VelocityLikelihood(MultivariateGaussianLikelihood):
#     def __init__(self, likelihood=None, model_name=None, covariance=None, vector=None):
#         self.vector = vector
#         self.covariance = covariance
#         self.model_name = model_name


# class DensityLikelihood(MultivariateGaussianLikelihood):
#     def __init__(self, function=None, rsd=None):
#         self.function = function
#         self.rsd = rsd

#     def get_log_like_density(
#         self, fitpar, density, density_err, cov_gg_b2, cov_gg_bf, cov_gg_f2, m_index_gg
#     ):
#         fs8 = fitpar[0]
#         bs8 = fitpar[1]
#         sig_g = fitpar[2]

#         # Build cov matrix
#         cov_gg_b2_all = np.sum(
#             np.array([cov_gg_b2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
#             axis=0,
#         )
#         cov_gg_bf_all = np.sum(
#             np.array([cov_gg_bf[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
#             axis=0,
#         )
#         cov_gg_f2_all = np.sum(
#             np.array([cov_gg_f2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
#             axis=0,
#         )
#         cov_matrix = (
#             bs8**2 * cov_gg_b2_all
#             + bs8 * fs8 * cov_gg_bf_all
#             + fs8**2 * cov_gg_f2_all
#         )

#         # Add shot noise
#         diag_add = density_err**2
#         cov_matrix += np.diag(diag_add)

#         log_like = self.log_likelihood(density, cov_matrix)
#         return log_like

#     def get_log_like_density_no_rsd(
#         self, fitpar, density, density_err, cov_gg_b2, m_index_gg
#     ):
#         bs8 = fitpar[0]
#         sig_g = fitpar[1]

#         # Build cov matrix
#         cov_gg_b2_all = np.sum(
#             np.array([cov_gg_b2[i] * sig_g**m for i, m in enumerate(m_index_gg)]),
#             axis=0,
#         )
#         cov_matrix = bs8**2 * cov_gg_b2_all

#         # Add shot noise
#         diag_add = density_err**2
#         cov_matrix += np.diag(diag_add)

#         log_like = self.log_likelihood(density, cov_matrix)
#         return log_like

#     @classmethod
#     def init_from_matrix(cls):
#         return cls.init_from_property_files()


# class JointLikelihood(MultivariateGaussianLikelihood):
#     def __init__(self, likelihood=None, model_name=None, covariance=None, vector=None):
#         self.vector = vector
#         self.covariance = covariance
#         self.model_name = model_name
