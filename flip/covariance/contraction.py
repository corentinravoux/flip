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
        los_definition=None,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.contraction_dict = contraction_dict
        self.coordinates_dict = coordinates_dict
        self.basis_definition = basis_definition
        self.los_definition = los_definition

    @classmethod
    def init_from_flip(
        cls,
        model_name,
        model_type,
        power_spectrum_dict,
        r_perpendicular,
        r_parallel,
        r_reference_perpendicular,
        r_reference_parallel,
        additional_parameters_values=None,
        basis_definition="bisector",
        los_definition="bisector",
        **kwargs,
    ):
        """
        The init_contraction_from_flip function is a helper function that allows the user to initialize
        a Contraction object from an existing FLIP object. This is useful for when you want to use the same
        FLIP object multiple times, but with different contraction parameters. For example, if you wanted to
        contract a covariance matrix at two different values of r_perpendicular and r_parallel (e.g., one set of values for calculating the power spectrum and another set of values for calculating correlation functions), then this function would be helpful.

        Args:
            cls: Create an instance of the class that called this function
            model_name: Define the model name
            model_type: Determine the type of model,
            power_spectrum_dict: Pass the power spectrum to the contraction_flip function
            r_perpendicular: Define the perpendicular distance from the reference point
            r_parallel: Define the parallel distance at which to evaluate the correlation function
            r_reference_perpendicular: Define the reference point for the perpendicular distance
            r_reference_parallel: Set the reference parallel coordinate
            additional_parameters_values: Pass in the values of the additional parameters
            basis_definition: Define the basis of the contraction
            los_definition: Define the line of sight
            **kwargs: Pass keyword arguments to the function
            : Define the model type

        Returns:
            An instance of the contraction class
        """
        (
            contraction_covariance_dict,
            contraction_coordinates_dict,
        ) = contract_covariance(
            model_name,
            model_type,
            power_spectrum_dict,
            r_perpendicular,
            r_parallel,
            r_reference_perpendicular,
            r_reference_parallel,
            additional_parameters_values=additional_parameters_values,
            basis_definition=basis_definition,
            los_definition=los_definition,
            **kwargs,
        )

        return cls(
            model_name=model_name,
            model_type=model_type,
            contraction_covariance_dict=contraction_covariance_dict,
            contraction_coordinates_dict=contraction_coordinates_dict,
            contraction_basis_definition=basis_definition,
            contraction_los_definition=los_definition,
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
        coefficients_dict = eval(f"coefficients_{self.model_name}.get_coefficients")(
            self.model_type,
            parameter_values_dict,
        )
        contraction_covariance_sum_dict = {}
        if self.model_type == "density":
            contraction_covariance_sum_dict["gg"] = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.contraction_covariance_dict["gg"])
                ],
                axis=0,
            )

        elif self.model_type == "velocity":
            contraction_covariance_sum_dict["vv"] = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.contraction_covariance_dict["vv"])
                ],
                axis=0,
            )

        elif self.model_type in ["density_velocity", "full"]:
            if self.model_type == "full":
                contraction_covariance_sum_dict["gv"] = np.sum(
                    [
                        coefficients_dict["gv"][i] * cov
                        for i, cov in enumerate(self.contraction_covariance_dict["gv"])
                    ],
                    axis=0,
                )
            contraction_covariance_sum_dict["gg"] = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.contraction_covariance_dict["gg"])
                ],
                axis=0,
            )

            contraction_covariance_sum_dict["vv"] = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.contraction_covariance_dict["vv"])
                ],
                axis=0,
            )
        else:
            log.add(f"Wrong model type in the loaded covariance.")

        return contraction_covariance_sum_dict


def compute_contraction_coordinates(
    r_perpendicular,
    r_parallel,
    r_reference_perpendicular,
    r_reference_parallel,
    basis_definition,
    los_definition,
):
    coord_rper_rpar = np.array(
        np.meshgrid(r_perpendicular, r_parallel, indexing="ij")
    ).reshape((2, len(r_perpendicular) * len(r_parallel)))

    r_reference = np.sqrt(r_reference_perpendicular**2 + r_reference_parallel**2)
    r = np.sqrt(coord_rper_rpar[0, :] ** 2 + coord_rper_rpar[1, :] ** 2)

    if basis_definition == "bisector":
        # r_perp, r_par and phi are defined with respect to the bisector between the two points.
        r_1 = np.sqrt(
            (r_reference_perpendicular + coord_rper_rpar[0, :]) ** 2
            + (r_reference_parallel + coord_rper_rpar[1, :]) ** 2
        )

        phi = np.arccos(np.clip(coord_rper_rpar[1, :] / r, -1.0, 1.0))
        theta = 2 * np.arcsin(np.clip(r * np.sin(phi) / (r_reference + r_1), -1.0, 1.0))

    elif basis_definition == "mean":
        # r_perp, r_par and phi are defined with respect to the mean between the two points.
        r_1 = np.sqrt(
            (r_reference_perpendicular + coord_rper_rpar[0, :]) ** 2
            + (r_reference_parallel + coord_rper_rpar[1, :]) ** 2
        )

        phi = np.arccos(np.clip(coord_rper_rpar[1, :] / r, -1.0, 1.0))
        theta = np.arcsin(
            np.clip(r * np.sin(phi) / (2 * r_reference), -1.0, 1.0)
        ) + np.arcsin(np.clip(r * np.sin(phi) / (2 * r_1), -1.0, 1.0))
    elif basis_definition == "endpoint":
        # r_perp, r_par are defined with respect to r_reference. phi can be defined by mean or bisector.
        theta = np.arctan2(coord_rper_rpar[0, :], r_reference + coord_rper_rpar[1, :])
        if los_definition == "bisector":
            phi = np.arcsin(
                np.clip(
                    ((r_reference / r) + (coord_rper_rpar[0, :] / (r * np.sin(theta))))
                    * np.sin(theta / 2),
                    -1.0,
                    1.0,
                )
            )
        elif los_definition == "mean":
            phi = np.arccos(np.clip(r**2 / 2 + r_parallel * r_reference, -1.0, 1.0))

    coordinates = np.zeros((3, len(r_perpendicular) * len(r_parallel)))
    coordinates[0, :] = r
    coordinates[1, :] = theta
    coordinates[2, :] = phi

    return coord_rper_rpar, coordinates


def contract_covariance(
    model_name,
    model_type,
    power_spectrum_dict,
    r_perpendicular,
    r_parallel,
    r_reference_perpendicular,
    r_reference_parallel,
    additional_parameters_values=None,
    basis_definition="bisector",
    los_definition="bisector",
    number_worker=8,
    hankel=True,
):
    coord_rper_rpar, coordinates = compute_contraction_coordinates(
        r_perpendicular,
        r_parallel,
        r_reference_perpendicular,
        r_reference_parallel,
        basis_definition,
        los_definition,
    )
    contraction_covariance_dict = {}
    if model_type in ["density", "full", "density_velocity"]:
        contraction_covariance_dict["gg"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "gg",
            power_spectrum_dict["gg"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:].reshape(-1, len(r_perpendicular), len(r_parallel))

    if model_type in ["velocity", "full", "density_velocity"]:
        contraction_covariance_dict["vv"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "vv",
            power_spectrum_dict["vv"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:].reshape(-1, len(r_perpendicular), len(r_parallel))

    if model_type == "full":
        contraction_covariance_dict["gv"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "gv",
            power_spectrum_dict["gv"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:].reshape(-1, len(r_perpendicular), len(r_parallel))

    contraction_coordinates_dict = {
        "r_perpendicular": coord_rper_rpar[0].reshape(
            len(r_perpendicular), len(r_parallel)
        ),
        "r_parallel": coord_rper_rpar[1].reshape(len(r_perpendicular), len(r_parallel)),
        "r": coordinates[0].reshape(len(r_perpendicular), len(r_parallel)),
        "theta": coordinates[1].reshape(len(r_perpendicular), len(r_parallel)),
        "phi": coordinates[2].reshape(len(r_perpendicular), len(r_parallel)),
    }
    return contraction_covariance_dict, contraction_coordinates_dict
