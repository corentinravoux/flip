import importlib

import numpy as np

from flip.covariance import generator as generator_flip
from flip.utils import create_log

log = create_log()


class Contraction:
    """Container for contracted covariance components and coordinates.

    Holds precomputed contraction tensors (per block gg/gv/vv) and the
    corresponding coordinate grids. Provides utilities to build from flip
    covariance generators and to evaluate weighted sums via model coefficients.
    """

    def __init__(
        self,
        model_name=None,
        model_kind=None,
        contraction_dict=None,
        coordinates_dict=None,
        basis_definition=None,
        variant=None,
    ):
        """Initialize the contraction container.

        Args:
            model_name (str|None): Covariance model package name.
            model_kind (str|None): `"density"`, `"velocity"`, `"full"`, or `"density_velocity"`.
            contraction_dict (dict|None): Arrays per block (e.g., `{"gg": [..], "vv": [..]}`).
            coordinates_dict (dict|None): Coordinates like `r_perpendicular`, `r_parallel`, etc.
            basis_definition (str|None): Basis choice, e.g., `"bisector"`, `"mean"`, `"endpoint"`.
            variant (str|None): Optional model variant.
        """
        self.model_name = model_name
        self.model_kind = model_kind
        self.contraction_dict = contraction_dict
        self.coordinates_dict = coordinates_dict
        self.basis_definition = basis_definition
        self.variant = variant

    @classmethod
    def init_from_flip(
        cls,
        model_name,
        model_kind,
        power_spectrum_dict,
        coord_1,
        coord_2,
        coord_1_reference,
        coord_2_reference,
        coordinate_type="rprt",
        additional_parameters_values=None,
        basis_definition="bisector",
        redshift=None,
        variant=None,
        **kwargs,
    ):
        """Build a `Contraction` from flip covariance generator outputs.

        Args:
            model_name (str): Covariance model name under `flip.covariance`.
            model_kind (str): `"density"`, `"velocity"`, `"full"`, or `"density_velocity"`.
            power_spectrum_dict (dict): Power spectra per block: keys `gg`, `vv`, and optional `gv`.
            coord_1 (ndarray): First coordinate grid (e.g., `r_perp` or `r`).
            coord_2 (ndarray): Second coordinate grid (e.g., `r_par` or `mu`).
            coord_1_reference (float): Reference point first coordinate.
            coord_2_reference (float): Reference point second coordinate.
            coordinate_type (str): `"rprt"` or `"rmu"`.
            additional_parameters_values (dict|None): Extra model parameters.
            basis_definition (str): Basis choice for angular definitions.
            redshift (float|None): Optional redshift context.
            variant (str|None): Model variant.
            **kwargs: Options forwarded to generator.

        Returns:
            Contraction: Initialized instance with tensors and coordinates.
        """
        (
            contraction_dict,
            coordinates_dict,
        ) = contract_covariance(
            model_name,
            model_kind,
            power_spectrum_dict,
            coord_1,
            coord_2,
            coord_1_reference,
            coord_2_reference,
            coordinate_type=coordinate_type,
            additional_parameters_values=additional_parameters_values,
            basis_definition=basis_definition,
            redshift=redshift,
            **kwargs,
        )

        return cls(
            model_name=model_name,
            model_kind=model_kind,
            contraction_dict=contraction_dict,
            coordinates_dict=coordinates_dict,
            basis_definition=basis_definition,
            variant=variant,
        )

    @property
    def type(self):
        """Return the model kind and log a short description.

        Returns:
            str: `model_kind` as provided on initialization.
        """
        if self.model_kind == "velocity":
            log.add("The covariance model is computed for velocity")
        elif self.model_kind == "density":
            log.add("The covariance model is computed for density")
        elif self.model_kind == "density_velocity":
            log.add(
                "The covariance model is computed for velocity and density, without cross-term"
            )
        elif self.model_kind == "full":
            log.add(
                "The covariance model is computed for velocity and density, with cross-term"
            )
        return self.model_kind

    def compute_contraction_sum(
        self,
        parameter_values_dict,
    ):
        """Compute weighted sum of contraction tensors using model coefficients.

        Args:
            parameter_values_dict (dict): Parameters to obtain coefficients.

        Returns:
            dict: Sum per block, e.g., `{"gg": array, "vv": array, "gv": array}`.
        """
        coefficients = importlib.import_module(
            f"flip.covariance.analytical.{self.model_name}.coefficients"
        )

        coefficients_dict = coefficients.get_coefficients(
            parameter_values_dict,
            self.model_kind,
            variant=self.variant,
        )
        contraction_covariance_sum_dict = {}
        if self.model_kind == "density":
            contraction_covariance_sum_dict["gg"] = np.sum(
                [
                    coefficients_dict["gg"][i] * cov
                    for i, cov in enumerate(self.contraction_dict["gg"])
                ],
                axis=0,
            )

        elif self.model_kind == "velocity":
            contraction_covariance_sum_dict["vv"] = np.sum(
                [
                    coefficients_dict["vv"][i] * cov
                    for i, cov in enumerate(self.contraction_dict["vv"])
                ],
                axis=0,
            )

        elif self.model_kind in ["density_velocity", "full"]:
            if self.model_kind == "full":
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
            log.add("Wrong model type in the loaded covariance.")

        return contraction_covariance_sum_dict


def compute_contraction_coordinates(
    coord_1,
    coord_2,
    coord_1_reference,
    coord_2_reference,
    coordinate_type,
    basis_definition,
):
    """Compute coordinate grids and derived angles for contractions.

    Supports `rmu` and `rprt` input coordinate parameterizations and basis
    definitions `bisector`, `mean`, and `endpoint`.

    Args:
        coord_1 (ndarray): First coordinate grid.
        coord_2 (ndarray): Second coordinate grid.
        coord_1_reference (float): Reference first coordinate.
        coord_2_reference (float): Reference second coordinate.
        coordinate_type (str): `"rmu"` or `"rprt"`.
        basis_definition (str): `"bisector"`, `"mean"`, or `"endpoint"`.

    Returns:
        tuple: `(coordinates_dict, coordinates)` where `coordinates_dict` holds
        2D grids and `coordinates` is a stacked array `[r, theta, phi]` per point.
    """
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

    phi = np.arccos(np.clip(coord_rper_rpar[1, :] / r, -1.0, 1.0))
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

        theta = np.arcsin(
            np.clip(r * np.sin(phi) / (2 * r_reference), -1.0, 1.0)
        ) + np.arcsin(np.clip(r * np.sin(phi) / (2 * r_1), -1.0, 1.0))
    elif basis_definition == "endpoint":
        # r_perp, r_par are defined with respect to r_reference = d.
        theta = np.arctan2(coord_rper_rpar[0, :], r_reference + coord_rper_rpar[1, :])

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
    model_kind,
    power_spectrum_dict,
    coord_1,
    coord_2,
    coord_1_reference,
    coord_2_reference,
    coordinate_type="rprt",
    additional_parameters_values=None,
    basis_definition="bisector",
    redshift=None,
    number_worker=8,
    hankel=True,
):
    """Generate contraction tensors for the specified model and blocks.

    Args:
        model_name (str): Covariance model name under `flip.covariance`.
        model_kind (str): `"density"`, `"velocity"`, `"full"`, or `"density_velocity"`.
        power_spectrum_dict (dict): Power spectra per block.
        coord_1 (ndarray): First coordinate grid.
        coord_2 (ndarray): Second coordinate grid.
        coord_1_reference (float): Reference first coordinate.
        coord_2_reference (float): Reference second coordinate.
        coordinate_type (str): Input parameterization, `"rprt"` or `"rmu"`.
        additional_parameters_values (dict|None): Extra model parameters.
        basis_definition (str): Basis choice.
        redshift (float|None): Optional redshift context.
        number_worker (int): Parallel workers used by generator.
        hankel (bool): Use FFTLog Hankel transform when True.

    Returns:
        tuple: `(contraction_dict, coordinates_dict)`.
    """
    # r_perpendicular : coord_1, r_parallel : coord_2
    coordinates_dict, coordinates = compute_contraction_coordinates(
        coord_1,
        coord_2,
        coord_1_reference,
        coord_2_reference,
        coordinate_type,
        basis_definition,
    )

    contraction_dict = {}
    if model_kind in ["density", "full", "density_velocity"]:
        contraction_dict["gg"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "gg",
            power_spectrum_dict["gg"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:].reshape(-1, len(coord_1), len(coord_2))

    if model_kind in ["velocity", "full", "density_velocity"]:
        contraction_dict["vv"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "vv",
            power_spectrum_dict["vv"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, 1:].reshape(-1, len(coord_1), len(coord_2))

    if model_kind == "full":
        contraction_dict["gv"] = generator_flip.compute_coeficient(
            [coordinates],
            model_name,
            "gv",
            power_spectrum_dict["gv"],
            additional_parameters_values=additional_parameters_values,
            number_worker=number_worker,
            hankel=hankel,
        )[:, :].reshape(-1, len(coord_1), len(coord_2))

    return contraction_dict, coordinates_dict
