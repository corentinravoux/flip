import importlib
import time

import numpy as np

from flip.utils import create_log

log = create_log()


class FisherMatrix:
    def __init__(
        self,
        covariance=None,
        fisher_matrix=None,
    ):
        self.covariance = covariance
        self.fisher_matrix = fisher_matrix

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        parameter_values_dict,
        **kwargs,
    ):

        # CR - Fisher matrix from https://arxiv.org/pdf/astro-ph/9603021 to implement

        vector_error = load_error_vector(
            covariance.model_type,
            parameter_values_dict,
        )

        covariance_sum = covariance.compute_covariance_sum(
            parameter_values_dict, vector_error
        )
        covariance_coefficients, covariance_derivatives = (
            cls.compute_covariance_derivatives(
                covariance,
                parameter_values_dict,
            )
        )

        A_matrices = []
        for derivative in enumerate(covariance_derivatives):
            A_matrices.append(np.linalg.inv(covariance_sum) * derivative)

        fisher_matrix = np.zeros((len(A_matrices), len(A_matrices)))
        for i in range(len(A_matrices)):
            for j in range(len(A_matrices)):
                fisher_matrix[i][j] = 0.5 * np.trace(
                    np.dot(A_matrices[i], A_matrices[j])
                )

        return cls(
            covariance=covariance,
            fisher_matrix=fisher_matrix,
        )

    @classmethod
    def compute_covariance_derivatives(
        cls,
        covariance,
        parameter_values_dict,
    ):
        coefficients = importlib.import_module(
            f"flip.covariance.{covariance.model_name}.coefficients"
        )

        coefficients_dict = coefficients.get_coefficients(
            covariance.model_type,
            parameter_values_dict,
            variant=covariance.variant,
        )
        if covariance.model_type == "density":
            return [coefficients_dict["gg"]], [covariance.covariance_dict["gg"]]
        elif covariance.model_type == "velocity":
            return [coefficients_dict["vv"]], [covariance.covariance_dict["vv"]]
        elif covariance.model_type == "density_velocity":
            return [coefficients_dict["gg"], coefficients_dict["vv"]], [
                covariance.covariance_dict["gg"],
                covariance.covariance_dict["vv"],
            ]
        elif covariance.model_type == "full":
            return [
                coefficients_dict["gg"],
                coefficients_dict["gv"],
                coefficients_dict["vv"],
            ], [
                covariance.covariance_dict["gg"],
                covariance.covariance_dict["gv"],
                covariance.covariance_dict["vv"],
            ]
