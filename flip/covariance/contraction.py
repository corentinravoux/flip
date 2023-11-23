import numpy as np

from flip.covariance import generator as generator_flip


def compute_contraction_coordinates(
    r_perpendicular,
    r_parallel,
    r_reference_perpendicular,
    r_reference_parallel,
    basis_definition,
    angle_definition,
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

    elif basis_definition == "midpoint":
        # r_perp, r_par and phi are defined with respect to the midpoint between the two points.
        r_1 = np.sqrt(
            (r_reference_perpendicular + coord_rper_rpar[0, :]) ** 2
            + (r_reference_parallel + coord_rper_rpar[1, :]) ** 2
        )

        phi = np.arccos(np.clip(coord_rper_rpar[1, :] / r, -1.0, 1.0))
        theta = np.arcsin(
            np.clip(r * np.sin(phi) / (2 * r_reference), -1.0, 1.0)
        ) + np.arcsin(np.clip(r * np.sin(phi) / (2 * r_1), -1.0, 1.0))
    elif basis_definition == "reference":
        # r_perp, r_par and phi are defined with respect to r_reference.
        theta = np.arctan2(coord_rper_rpar[0, :], r_reference + coord_rper_rpar[1, :])
        if angle_definition == "bisector":
            phi = np.arcsin(
                np.clip(
                    ((r_reference / r) + (coord_rper_rpar[0, :] / (r * np.sin(theta))))
                    * np.sin(theta / 2),
                    -1.0,
                    1.0,
                )
            )
        elif angle_definition == "midpoint":
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
    basis_definition="middle",
    angle_definition="bisector",
    number_worker=8,
    hankel=True,
):
    coord_rper_rpar, coordinates = compute_contraction_coordinates(
        r_perpendicular,
        r_parallel,
        r_reference_perpendicular,
        r_reference_parallel,
        basis_definition,
        angle_definition,
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
