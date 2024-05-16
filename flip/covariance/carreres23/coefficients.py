import numpy as np


def get_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
):
    coefficients_dict = {}
    coefficients_dict["vv"] = [parameter_values_dict["fs8"] ** 2]
    return coefficients_dict


def get_diagonal_coefficients(model_type, parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict


def get_partial_derivative_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
):
    partial_coefficients_dict = {}
    if variant == "growth_index":
        partial_coefficients_dict = {
            "Omegam": {
                "vv": [
                    2
                    * parameter_values_dict["gamma"]
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"] - 1)
                    * parameter_values_dict["s8"] ** 2
                ]
            },
            "gamma": {
                "vv": [
                    2
                    * np.log(parameter_values_dict["Omegam"])
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"] ** 2
                ]
            },
            "s8": {
                "vv": [
                    2
                    * parameter_values_dict["Omegam"]
                    ** (2 * parameter_values_dict["gamma"])
                    * parameter_values_dict["s8"],
                ]
            },
        }
    else:
        partial_coefficients_dict = {"fs8": {"vv": [2 * parameter_values_dict["fs8"]]}}
    return partial_coefficients_dict
