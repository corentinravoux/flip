import numpy as np


def get_partial_derivative_coefficients(
    model_kind,
    parameter_values_dict,
    variant=None,
    covariance_prefactor_dict=None,
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
        }
    else:
        partial_coefficients_dict = {
            "fs8": {
                "vv": [
                    2 * parameter_values_dict["fs8"],
                ],
            },
        }
    return partial_coefficients_dict
