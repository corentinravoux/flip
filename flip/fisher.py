import importlib
import time

import numpy as np

from flip.utils import create_log

log = create_log()


def inverse_covariance_inverse(covariance):
    return np.linalg.inv(covariance)


class FisherMatrix:

    _default_fisher_properties = {
        "inversion_method": "inverse",
        "negative_log_likelihood": True,
    }

    def __init__(
        self,
        covariance=None,
        inverse_covariance_sum=None,
    ):

        self.covariance = covariance
        self.inverse_covariance_sum = inverse_covariance_sum

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_values_dict,
        fisher_properties={},
    ):
        if covariance.full_matrix is False:
            covariance.compute_full_matrix()

        fisher_properties = {
            **cls._default_fisher_properties,
            **fisher_properties,
        }

        vector_error = data(parameter_values_dict)

        covariance_sum = covariance.compute_covariance_sum(
            parameter_values_dict, vector_error
        )

        inverse_covariance_sum = eval(
            f"inverse_covariance_{fisher_properties['inversion_method']}"
        )(covariance_sum)

        return cls(
            covariance=covariance,
            inverse_covariance_sum=inverse_covariance_sum,
        )


    def compute_covariance_derivative(
        self,
        partial_coefficients_dict_param,
    ):

        if self.covariance.model_type == "density":
            covariance_derivative_sum = np.sum(
                [
                    partial_coefficients_dict_param["gg"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["gg"])
                ],
                axis=0,
            )

        elif self.covariance.model_type == "velocity":
            covariance_derivative_sum = np.sum(
                [
                    partial_coefficients_dict_param["vv"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["vv"])
                ],
                axis=0,
            )

        elif self.covariance.model_type in ["density_velocity", "full"]:
            number_densities = self.covariance.number_densities
            number_velocities = self.covariance.number_velocities

            if self.covariance.model_type == "density_velocity":
                covariance_derivative_sum_gv = np.zeros(
                    (number_densities, number_velocities)
                )
            elif self.covariance.model_type == "full":
                covariance_derivative_sum_gv = np.sum(
                    [
                        partial_coefficients_dict_param["gv"][i] * cov
                        for i, cov in enumerate(self.covariance.covariance_dict["gv"])
                    ],
                    axis=0,
                )
            covariance_derivative_sum_gg = np.sum(
                [
                    partial_coefficients_dict_param["gg"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["gg"])
                ],
                axis=0,
            )

            covariance_derivative_sum_vv = np.sum(
                [
                    partial_coefficients_dict_param["vv"][i] * cov
                    for i, cov in enumerate(self.covariance.covariance_dict["vv"])
                ],
                axis=0,
            )
            covariance_derivative_sum_vg = -covariance_derivative_sum_gv.T

            covariance_derivative_sum = np.block(
                [
                    [covariance_derivative_sum_gg, covariance_derivative_sum_gv],
                    [covariance_derivative_sum_vg, covariance_derivative_sum_vv],
                ]
            )
        else:
            log.add(f"Wrong model type in the loaded covariance.")

        return covariance_derivative_sum

    def compute_fisher_matrix(
        self,
        parameter_values_dict,
        variant=None,
    ):

        coefficients = importlib.import_module(
            f"flip.covariance.{self.covariance.model_name}.fisher_terms"
        )
        partial_coefficients_dict = coefficients.get_partial_derivative_coefficients(
            self.covariance.model_type,
            parameter_values_dict,
            variant=self.covariance.variant,
            redshift_dict=self.covariance.redshift_dict,
        )
        parameter_name_list = []
        covariance_derivative_sum_list = []

        for (
            parameter_name,
            partial_coefficients_dict_param,
        ) in partial_coefficients_dict.items():
            parameter_name_list.append(parameter_name)
            covariance_derivative_sum_list.append(
                np.dot(
                    self.inverse_covariance_sum,
                    self.compute_covariance_derivative(
                        partial_coefficients_dict_param,
                    ),
                )
            )

        fisher_matrix = np.zeros(
            (
                len(partial_coefficients_dict.keys()),
                len(partial_coefficients_dict.keys()),
            )
        )
        for i in range(len(fisher_matrix)):
            for j in range(i, len(fisher_matrix)):
                fisher_matrix[i][j] = 0.5 * np.trace(
                    np.dot(
                        covariance_derivative_sum_list[i],
                        covariance_derivative_sum_list[j],
                    )
                )
        for i in range(len(fisher_matrix)):
            for j in range(i):
                fisher_matrix[i][j] = fisher_matrix[j][i]

        return parameter_name_list, fisher_matrix
