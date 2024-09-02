import numpy as np


def get_partial_derivative_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
    redshift_dict=None,
):
    if model_type == "density":
        return get_partial_derivative_coefficients_density(
            parameter_values_dict,
            variant=variant,
        )
    elif model_type == "velocity":
        return get_partial_derivative_coefficients_velocity(
            parameter_values_dict,
            variant=variant,
        )
    elif model_type == "density_velocity":
        return get_partial_derivative_coefficients_density_velocity(
            parameter_values_dict,
            variant=variant,
        )
    elif model_type == "full":
        return get_partial_derivative_coefficients_full(
            parameter_values_dict,
            variant=variant,
        )


def get_partial_derivative_coefficients_velocity(
    parameter_values_dict,
    variant=None,
):
    if variant == "growth_index":
        partial_coefficients_dict = {
            "Omegam": {
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    / parameter_values_dict["Omegam"],
                ],
            },
            "gamma": {
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                ],
            },
            "s8": {
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"],
                ],
            },
            "bs8": {
                "vv": [
                    0,
                ],
            },
        }
    else:
        partial_coefficients_dict = {
            "fs8": {
                "vv": [
                    2 * parameter_values_dict["fs8"],
                ],
            },
            "bs8": {
                "vv": [
                    0,
                ],
            },
        }
    return partial_coefficients_dict


def get_partial_derivative_coefficients_density(
    parameter_values_dict,
    variant=None,
):
    if variant == "growth_index":
        partial_coefficients_dict = {
            "Omegam": {
                "gg": [
                    0,
                ],
            },
            "gamma": {
                "gg": [
                    0,
                ],
            },
            "s8": {
                "gg": [
                    0,
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                ],
            },
        }
    else:
        partial_coefficients_dict = {
            "fs8": {
                "gg": [
                    0,
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                ],
            },
        }
    return partial_coefficients_dict


def get_partial_derivative_coefficients_density_velocity(
    parameter_values_dict,
    variant=None,
):
    if variant == "growth_index":
        partial_coefficients_dict = {
            "Omegam": {
                "gg": [
                    0,
                ],
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    / parameter_values_dict["Omegam"],
                ],
            },
            "gamma": {
                "gg": [
                    0,
                ],
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                ],
            },
            "s8": {
                "gg": [
                    0,
                ],
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"],
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                ],
                "vv": [
                    0,
                ],
            },
        }
    else:
        partial_coefficients_dict = {
            "fs8": {
                "gg": [
                    0,
                ],
                "vv": [
                    2 * parameter_values_dict["fs8"],
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                ],
                "vv": [
                    0,
                ],
            },
        }
    return partial_coefficients_dict


def get_partial_derivative_coefficients_full(
    parameter_values_dict,
    variant=None,
):
    if variant == "growth_index":
        partial_coefficients_dict = {
            "Omegam": {
                "gg": [
                    0,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    / parameter_values_dict["Omegam"],
                ],
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    / parameter_values_dict["Omegam"],
                ],
            },
            "gamma": {
                "gg": [
                    0,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * np.log(parameter_values_dict["Omegam"]),
                ],
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                ],
            },
            "s8": {
                "gg": [
                    0,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"],
                ],
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"],
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"],
                ],
                "vv": [
                    0,
                ],
            },
        }
    else:
        partial_coefficients_dict = {
            "fs8": {
                "gg": [
                    0,
                ],
                "gv": [
                    parameter_values_dict["bs8"],
                ],
                "vv": [
                    2 * parameter_values_dict["fs8"],
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                ],
                "gv": [
                    parameter_values_dict["fs8"],
                ],
                "vv": [
                    0,
                ],
            },
        }
    return partial_coefficients_dict
