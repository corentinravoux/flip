def get_coefficients(
    parameter_values_dict,
    model_kind,
    variant=None,
    redshift_dict=None,
):
    coefficients_dict = {}
    if model_kind in ["density", "full", "density_velocity"]:
        if variant == "nobeta":
            coefficients_dict["gg"] = [
                parameter_values_dict["bs8"] ** 2,
                parameter_values_dict["bs8"] * parameter_values_dict["fs8"],
                parameter_values_dict["fs8"] ** 2,
            ]
        else:
            coefficients_dict["gg"] = [
                parameter_values_dict["bs8"] ** 2,
                parameter_values_dict["bs8"] ** 2 * parameter_values_dict["beta_f"],
                parameter_values_dict["bs8"] ** 2
                * parameter_values_dict["beta_f"] ** 2,
            ]
    if model_kind in ["full"]:
        if variant == "nobeta":
            coefficients_dict["gv"] = [
                parameter_values_dict["bs8"] * parameter_values_dict["fs8"],
                parameter_values_dict["fs8"] ** 2,
            ]
        else:
            coefficients_dict["gv"] = [
                parameter_values_dict["bs8"] * parameter_values_dict["fs8"],
                parameter_values_dict["bs8"]
                * parameter_values_dict["fs8"]
                * parameter_values_dict["beta_f"],
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
