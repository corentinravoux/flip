import numpy as np
from flip.covariance.lai22.flip_terms import dictionary_terms


def get_coefficients(model_type, parameter_values_dict):
    coefficients_dict = {}
    if model_type in ["density", "full", "density_velocity"]:
        coefficients_dict["gg"] = []
        gg_terms = dictionary_terms["gg"]
        sigg = parameter_values_dict["sigg"]
        for gg_term in gg_terms:
            term_index, m_index = np.array(gg_term.split("_")).astype(int)
            if term_index == 0:
                coefficients_dict["gg"].append(
                    parameter_values_dict["bs8"] ** 2 * sigg ** (2 * m_index)
                )
            elif term_index == 1:
                coefficients_dict["gg"].append(
                    parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * sigg ** (2 * m_index)
                )
            elif term_index == 2:
                coefficients_dict["gg"].append(
                    parameter_values_dict["fs8"] ** 2 * sigg ** (2 * m_index)
                )
    if model_type in ["full"]:
        coefficients_dict["gv"] = []
        gv_terms = dictionary_terms["gv"]
        sigg = parameter_values_dict["sigg"]
        for gv_term in gv_terms:
            term_index, m_index = np.array(gv_term.split("_")).astype(int)
            if term_index == 0:
                coefficients_dict["gv"].append(
                    parameter_values_dict["bs8"]
                    * parameter_values_dict["fs8"]
                    * sigg ** (2 * m_index)
                )
            elif term_index == 1:
                coefficients_dict["gv"].append(
                    parameter_values_dict["fs8"] ** 2 * sigg ** (2 * m_index)
                )
    if model_type in ["velocity", "full", "density_velocity"]:
        coefficients_dict["vv"] = [parameter_values_dict["fs8"] ** 2]
    return coefficients_dict


def get_diagonal_coefficients(model_type, parameter_values_dict):
    coefficients_dict = {}
    if model_type in ["density", "full", "density_velocity"]:
        coefficients_dict["gg"] = 0.0
    if model_type in ["velocity", "full", "density_velocity"]:
        coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict
