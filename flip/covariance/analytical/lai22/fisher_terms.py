import numpy as np


def get_partial_derivative_coefficients(
    model_kind,
    parameter_values_dict,
    variant=None,
    covariance_prefactor_dict=None,
):
    if model_kind == "density":
        return get_partial_derivative_coefficients_density(
            parameter_values_dict,
            variant=variant,
        )
    elif model_kind == "velocity":
        return get_partial_derivative_coefficients_velocity(
            parameter_values_dict,
            variant=variant,
        )
    elif model_kind == "density_velocity":
        return get_partial_derivative_coefficients_density_velocity(
            parameter_values_dict,
            variant=variant,
        )
    elif model_kind == "full":
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
            "beta_f": {
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "vv": [
                    0,
                ],
            },
        }
    elif variant == "growth_index_nobeta":
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
            "sigg": {
                "vv": [
                    0,
                ],
            },
        }
    elif variant == "nobeta":
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
            "sigg": {
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
            "beta_f": {
                "vv": [
                    0,
                ],
            },
            "sigg": {
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            "gamma": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            "s8": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["beta_f"] * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
            },
            "beta_f": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
            },
        }
    elif variant == "growth_index_nobeta":
        partial_coefficients_dict = {
            "Omegam": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 6
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 8
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 10
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 12
                    / parameter_values_dict["Omegam"],
                ],
            },
            "gamma": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 6
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 8
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 10
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 12
                    * np.log(parameter_values_dict["Omegam"]),
                ],
            },
            "s8": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
            },
        }
    elif variant == "nobeta":
        partial_coefficients_dict = {
            "fs8": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"],
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["fs8"],
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    parameter_values_dict["fs8"],
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 12,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
            },
        }
    else:
        partial_coefficients_dict = {
            "fs8": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["beta_f"] * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
            },
            "beta_f": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
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
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["beta_f"] * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "vv": [
                    0,
                ],
            },
            "beta_f": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
                "vv": [
                    0,
                ],
            },
        }
    elif variant == "growth_index_nobeta":
        partial_coefficients_dict = {
            "Omegam": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 6
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 8
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 10
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 12
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 6
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 8
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 10
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 12
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12,
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
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
                "vv": [
                    0,
                ],
            },
        }
    elif variant == "nobeta":
        partial_coefficients_dict = {
            "fs8": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"],
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["fs8"],
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "vv": [
                    2 * parameter_values_dict["fs8"],
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    parameter_values_dict["fs8"],
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 12,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "vv": [
                    2 * parameter_values_dict["fs8"],
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["beta_f"] * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "vv": [
                    0,
                ],
            },
            "beta_f": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    / parameter_values_dict["Omegam"],
                    0,
                    0,
                    0,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    * np.log(parameter_values_dict["Omegam"]),
                    0,
                    0,
                    0,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    0,
                    0,
                    0,
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
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["beta_f"] * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                ],
                "vv": [
                    0,
                ],
            },
            "beta_f": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "gv": [
                    0,
                    0,
                    0,
                    0,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                ],
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
                "gv": [
                    0,
                    2
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 5,
                    0,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                ],
                "vv": [
                    0,
                ],
            },
        }
    elif variant == "growth_index_nobeta":
        partial_coefficients_dict = {
            "Omegam": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 6
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 8
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 10
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 12
                    / parameter_values_dict["Omegam"],
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 2
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 4
                    / parameter_values_dict["Omegam"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 6
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 6
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 8
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 10
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 12
                    * np.log(parameter_values_dict["Omegam"]),
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 2
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 4
                    * np.log(parameter_values_dict["Omegam"]),
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 6
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"],
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
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
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 12,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "gv": [
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"],
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 6,
                    0,
                    0,
                    0,
                    0,
                ],
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
                "gv": [
                    0,
                    2
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"] ** parameter_values_dict["gamma"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["s8"]
                    * parameter_values_dict["sigg"] ** 5,
                    0,
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                ],
                "vv": [
                    0,
                ],
            },
        }
    elif variant == "nobeta":
        partial_coefficients_dict = {
            "fs8": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"],
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["fs8"],
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "gv": [
                    parameter_values_dict["bs8"],
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 6,
                    2 * parameter_values_dict["fs8"],
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 6,
                ],
                "vv": [
                    2 * parameter_values_dict["fs8"],
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    parameter_values_dict["fs8"],
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 12,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "gv": [
                    parameter_values_dict["fs8"],
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 6,
                    0,
                    0,
                    0,
                    0,
                ],
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
                "gv": [
                    0,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 5,
                    0,
                    2
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["fs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
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
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "gv": [
                    parameter_values_dict["bs8"],
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] * parameter_values_dict["sigg"] ** 6,
                    0,
                    0,
                    0,
                    0,
                ],
                "vv": [
                    2 * parameter_values_dict["fs8"],
                ],
            },
            "bs8": {
                "gg": [
                    2 * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2 * parameter_values_dict["beta_f"] * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "gv": [
                    parameter_values_dict["fs8"],
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["fs8"] * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"],
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["sigg"] ** 6,
                ],
                "vv": [
                    0,
                ],
            },
            "beta_f": {
                "gg": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    parameter_values_dict["bs8"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 8,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 10,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 12,
                ],
                "gv": [
                    0,
                    0,
                    0,
                    0,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 2,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 4,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 6,
                ],
                "vv": [
                    0,
                ],
            },
            "sigg": {
                "gg": [
                    0,
                    2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"]
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                    0,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                    8
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 7,
                    10
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 9,
                    12
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 11,
                ],
                "gv": [
                    0,
                    2
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * parameter_values_dict["sigg"] ** 5,
                    0,
                    2
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"],
                    4
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 3,
                    6
                    * parameter_values_dict["beta_f"] ** 2
                    * parameter_values_dict["bs8"] ** 2
                    * parameter_values_dict["sigg"] ** 5,
                ],
                "vv": [
                    0,
                ],
            },
        }
    return partial_coefficients_dict
