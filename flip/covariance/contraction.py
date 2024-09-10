import importlib

import numpy as np

from flip.covariance import generator as generator_flip
from flip.utils import create_log

log = create_log()


class Contraction:
    def __init__(
        self,
        model_name=None,
        model_type=None,
        contraction_dict=None,
        coordinates_dict=None,
        basis_definition=None,
        endpoint_los_definition=None,
        redshift_dict=None,
        variant=None,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.contraction_dict = contraction_dict
        self.coordinates_dict = coordinates_dict
        self.basis_definition = basis_definition
        self.endpoint_los_definition = endpoint_los_definition
        self.redshift_dict = redshift_dict
        self.variant = variant

    @classmethod
    def init_from_flip(
        cls,
        model_name,
        model_type,
        power_spectrum_dict,
        coord_1,
        coord_2,
        coord_1_reference,
        coord_2_reference,
        coordinate_type="rprt",
        additional_parameters_values=None,
        basis_definition="bisector",
        endpoint_los_definition="bisector",
        redshift=None,
        variant=None,
        **kwargs,
    ):
        (
            contraction_dict,
            coordinates_dict,
            redshift_dict,
        ) = contract_covariance(
            model_name,
            model_type,
            power_spectrum_dict,
            coord_1,
            coord_2,
            coord_1_reference,
            coord_2_reference,
            coordinate_type=coordinate_type,
            additional_parameters_values=additional_parameters_values,
            basis_definition=basis_definition,
            endpoint_los_definition=endpoint_los_definition,
            redshift=redshift,
            **kwargs,
        )

        return cls(
            model_name=model_name,
            model_type=model_type,
            contraction_dict=contraction_dict,
            coordinates_dict=coordinates_dict,
            basis_definition=basis_definition,
            endpoint_los_definition=endpoint_los_definition,
            redshift_dict=redshift_dict,
            variant=variant,
        )

    @property
    def type(self):
        """
        The type function is used to determine the type of covariance model that will be computed.
        The options are:
            - velocity: The covariance model is computed for velocity only.
            - density: The covariance model is computed for density only.
            - density_velocity: The covariance model is computed for both velocity and density, without cross-term (i.e., the covariances between velocities and densities are zero). This option should be used when computing a full 3D tomography in which we want to compute a separate 1D tomography along each axis (x, y, z

        Args:
            self: Represent the instance of the class

        Returns:
            The type of the model

        """
        if self.model_type == "velocity":
            log.add("The covariance model is computed for velocity")
        elif self.model_type == "density":
            log.add("The covariance model is computed for density")
        elif self.model_type == "density_velocity":
            log.add(
                "The covariance model is computed for velocity and density, without cross-term"
            )
        elif self.model_type == "full":
            log.add(
                "The covariance model is computed for velocity and density, with cross-term"
            )
        return self.model_type

    def compute_contraction_sum(
        self,
        parameter_values_dict,
    ):
        """
        The compute_contraction_sum function computes the sum of all the contractions
            for a given model type and parameter values.

        Args:
            self: Make the function a method of the class
            parameter_values_dict: Get the coefficients for each of the covariances
            : Get the coefficients of the model

        Returns:
            A dictionary of contraction_covariance_sum
        """
        coefficients = importlib.import_module(
            f"flip.covariance.{self.model_name}.coefficients"
        )

        coefficients_dict = coefficients.get_coefficients(
            self.model_type,
            parameter_values_dict,
            variant=self.variant,
            redshift_dict=self.redshift_dict,
        )
        contraction_covariance_sum_dict = {}
        if self.model_type == "density":
            contraction_covariance_sum_dict["gg"] = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.contraction_dict["gg"])
                ],
                axis=0,
            )

        elif self.model_type == "velocity":
            contraction_covariance_sum_dict["vv"] = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.contraction_dict["vv"])
                ],
                axis=0,
            )

        elif self.model_type in ["density_velocity", "full"]:
            if self.model_type == "full":
                contraction_covariance_sum_dict["gv"] = np.sum(
                    [
                        coefficients_dict["gv"][i] * cov
                        for i, cov in enumerate(self.contraction_dict["gv"])
                    ],
                    axis=0,
                )
            contraction_covariance_sum_dict["gg"] = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.contraction_dict["gg"])
                ],
                axis=0,
            )

            contraction_covariance_sum_dict["vv"] = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.contraction_dict["vv"])
                ],
                axis=0,
            )
        else:
            log.add(f"Wrong model type in the loaded covariance.")

        return contraction_covariance_sum_dict


def compute_contraction_coordinates(
    coord_1,
    coord_2,
    coord_1_reference,
    coord_2_reference,
    coordinate_type,
    basis_definition,
    endpoint_los_definition,
):
    shape_coord_1_coord_2 = len(coord_1) * len(coord_2)

    if coordinate_type == "rmu":
        # r = coord_1, mu = coord_2

        coord_rmu = np.array(np.meshgrid(coord_1, coord_2, indexing="ij")).reshape(
            (2, shape_coord_1_coord_2)
        )

        coord_rper_rpar = np.zeros((2, shape_coord_1_coord_2))

        coord_rper_rpar[0, :] = coord_rmu[0, :] * np.sqrt(1 - coord_rmu[1, :] ** 2)
        coord_rper_rpar[1, :] = coord_rmu[0, :] * coord_rmu[1, :]

        r_perpendicular_reference = coord_1_reference * np.sqrt(
            1 - coord_2_reference**2
        )
        r_parallel_reference = coord_1_reference * coord_2_reference

    elif coordinate_type == "rprt":
        # r_perpendicular = coord_1, r_parallel = coord_2

        coord_rper_rpar = np.array(
            np.meshgrid(coord_1, coord_2, indexing="ij")
        ).reshape((2, shape_coord_1_coord_2))

        r_perpendicular_reference = coord_1_reference
        r_parallel_reference = coord_2_reference

        coord_rmu = np.zeros((2, shape_coord_1_coord_2))

        coord_rmu[0, :] = np.sqrt(
            coord_rper_rpar[0, :] ** 2 + coord_rper_rpar[1, :] ** 2
        )
        coord_rmu[1, :] = coord_rper_rpar[1, :] / np.sqrt(
            coord_rper_rpar[0, :] ** 2 + coord_rper_rpar[1, :] ** 2
        )

    r_reference = np.sqrt(r_perpendicular_reference**2 + r_parallel_reference**2)
    r = np.sqrt(coord_rper_rpar[0, :] ** 2 + coord_rper_rpar[1, :] ** 2)

    if basis_definition == "bisector":
        # r_perp, r_par and phi are defined with respect to the bisector between the two points.
        r_1 = np.sqrt(
            (r_perpendicular_reference + coord_rper_rpar[0, :]) ** 2
            + (r_parallel_reference + coord_rper_rpar[1, :]) ** 2
        )

        phi = np.arccos(np.clip(coord_rper_rpar[1, :] / r, -1.0, 1.0))
        theta = 2 * np.arcsin(np.clip(r * np.sin(phi) / (r_reference + r_1), -1.0, 1.0))

    elif basis_definition == "mean":
        # r_perp, r_par and phi are defined with respect to the mean between the two points.
        r_1 = np.sqrt(
            (r_perpendicular_reference + coord_rper_rpar[0, :]) ** 2
            + (r_parallel_reference + coord_rper_rpar[1, :]) ** 2
        )

        phi = np.arccos(np.clip(coord_rper_rpar[1, :] / r, -1.0, 1.0))
        theta = np.arcsin(
            np.clip(r * np.sin(phi) / (2 * r_reference), -1.0, 1.0)
        ) + np.arcsin(np.clip(r * np.sin(phi) / (2 * r_1), -1.0, 1.0))
    elif basis_definition == "endpoint":
        # r_perp, r_par are defined with respect to r_reference. phi can be defined by mean or bisector.
        theta = np.arctan2(coord_rper_rpar[0, :], r_reference + coord_rper_rpar[1, :])
        if endpoint_los_definition == "bisector":
            phi = np.arcsin(
                np.clip(
                    ((r_reference / r) + (coord_rper_rpar[0, :] / (r * np.sin(theta))))
                    * np.sin(theta / 2),
                    -1.0,
                    1.0,
                )
            )
        elif endpoint_los_definition == "mean":
            phi = np.arccos(
                np.clip(r**2 / 2 + coord_rper_rpar[1, :] * r_reference, -1.0, 1.0)
            )

    coordinates = np.zeros((3, shape_coord_1_coord_2))
    coordinates[0, :] = r
    coordinates[1, :] = theta
    coordinates[2, :] = phi

    coordinates_dict = {
        "r_perpendicular": coord_rper_rpar[0].reshape(len(coord_1), len(coord_2)),
        "r_parallel": coord_rper_rpar[1].reshape(len(coord_1), len(coord_2)),
        "r": r.reshape(len(coord_1), len(coord_2)),
        "mu": coord_rmu[1, :].reshape(len(coord_1), len(coord_2)),
        "theta": theta.reshape(len(coord_1), len(coord_2)),
        "phi": phi.reshape(len(coord_1), len(coord_2)),
    }

    return coordinates_dict, coordinates


def contract_covariance(
    model_name,
    model_type,
    power_spectrum_dict,
    coord_1,
    coord_2,
    coord_1_reference,
    coord_2_reference,
    coordinate_type="rprt",
    additional_parameters_values=None,
    basis_definition="bisector",
    endpoint_los_definition="bisector",
    redshift=None,
    number_worker=8,
    hankel=True,
):
    # r_perpendicular : coord_1, r_parallel : coord_2
    coordinates_dict, coordinates = compute_contraction_coordinates(
        coord_1,
        coord_2,
        coord_1_reference,
        coord_2_reference,
        coordinate_type,
        basis_definition,
        endpoint_los_definition,
    )
    contraction_dict = {}
    if model_type in ["density", "full", "density_velocity"]:
        contraction_dict["gg"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "gg",
            power_spectrum_dict["gg"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:].reshape(-1, len(coord_1), len(coord_2))

    if model_type in ["velocity", "full", "density_velocity"]:
        contraction_dict["vv"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "vv",
            power_spectrum_dict["vv"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:].reshape(-1, len(coord_1), len(coord_2))

    if model_type == "full":
        contraction_dict["gv"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "gv",
            power_spectrum_dict["gv"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, :].reshape(-1, len(coord_1), len(coord_2))
    redshift_dict = generator_flip.generate_redshift_dict(
        model_name,
        model_type,
        redshift_velocity=redshift,
        redshift_density=redshift,
    )
    return contraction_dict, coordinates_dict, redshift_dict
