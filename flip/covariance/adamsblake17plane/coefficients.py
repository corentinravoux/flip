def get_coefficients(
    model_type,
    parameter_values_dict,
    variant=None,
):
    coefficients_dict = {}
    if model_type in ["density", "full", "density_velocity"]:
        coefficients_dict["gg"] = [
            parameter_values_dict["bs8"] ** 2,
        ]
    if model_type in ["full"]:
        coefficients_dict["gv"] = [
            parameter_values_dict["bs8"] * parameter_values_dict["fs8"],
        ]
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
