import importlib
import time

import numpy as np

from flip import vectors
from flip.utils import create_log

log = create_log()


class FisherMatrix:
    def __init__(
        self,
        covariance=None,
        data=None,
        fisher_matrix=None,
    ):
        self.covariance = covariance
        self.fisher_matrix = fisher_matrix

    @classmethod
    def init_from_covariance(
        cls,
        covariance,
        data,
        parameter_values_dict,
        fisher_properties,
        **kwargs,
    ):

        vector_error = cls.load_error_vector(
            covariance.model_type,
            data,
            parameter_values_dict,
            fisher_properties,
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

    def compute_covariance_derivatives(
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

    @classmethod
    def load_error_vector(
        cls,
        model_type,
        data,
        parameter_values_dict,
        fisher_properties,
    ):
        if model_type in ["velocity", "density_velocity", "full"]:
            velocity_error = vectors.load_velocity_error(
                data,
                parameter_values_dict,
                velocity_type=fisher_properties["velocity_type"],
                velocity_estimator=fisher_properties["velocity_estimator"],
            )

        if model_type in ["density", "density_velocity", "full"]:
            density_error = vectors.load_density_error(data)

        if model_type == "density":
            return density_error
        elif model_type == "velocity":
            return velocity_error
        elif model_type in ["density_velocity", "full"]:
            return np.concatenate([density_error, velocity_error], axis=0)
        else:
            log.add(f"Wrong model type in the loaded covariance.")
