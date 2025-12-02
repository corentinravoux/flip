def get_coefficients(
    parameter_values_dict,
    model_kind,
    variant=None,
    covariance_prefactor_dict=None,
):
    coefficients_dict = {}
    if model_kind in ["density", "full", "density_velocity"]:
        coefficients_dict["gg"] = [
            parameter_values_dict["bs8"] ** 2,
        ]
    if model_kind in ["full"]:
        coefficients_dict["gv"] = [
            parameter_values_dict["bs8"] * parameter_values_dict["fs8"],
        ]
    if model_kind in ["velocity", "full", "density_velocity"]:
        coefficients_dict["vv"] = [parameter_values_dict["fs8"] ** 2]
    return coefficients_dict


def get_diagonal_coefficients(parameter_values_dict, model_kind):
    coefficients_dict = {}
    if model_kind in ["density", "full", "density_velocity"]:
        coefficients_dict["gg"] = 0.0
    if model_kind in ["velocity", "full", "density_velocity"]:
        coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict
