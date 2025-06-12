import numpy as np


def get_coefficients(
    parameter_values_dict,
    model_kind,
    variant=None,
    redshift_dict=None,
):
    coefficients_dict = {}
    coefficients_dict["vv"] = [parameter_values_dict["fs8"] ** 2]
    return coefficients_dict


def get_diagonal_coefficients(parameter_values_dict, model_kind):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict
