import importlib

from flip.covariance import cov_utils
from flip.covariance.emulators.gpmatrix import evaluation as evaluation_gpmatrix
from flip.covariance.emulators.gpmatrix import training as training_gpmatrix
from flip.utils import create_log

log = create_log()

_avail_emulator_models = [
    "gpmatrix",
]


def compute_cov(
    emulator_model_name,
    covariance_list,
    parameter_values,
    covariance_type,
    test_emulator=False,
    **kwargs,
):

    if emulator_model_name not in _avail_emulator_models:
        log.add(
            f"Model {emulator_model_name} not available."
            f"Please choose between: {_avail_emulator_models}"
        )

    emulator_type = importlib.import_module(
        f"flip.covariance.emulators.{emulator_model_name}"
    )._emulator_type
    # Only implemented for matrix emulators, so far.

    training_module = eval(f"training_{emulator_model_name}")
    evaluation_module = eval(f"evaluation_{emulator_model_name}")

    models = training_module.train_model(
        covariance_list,
        parameter_values,
        covariance_type,
        **kwargs,
    )

    evaluation_functions = evaluation_module.return_evaluation_functions(
        models,
        **kwargs,
    )

    if test_emulator:
        evaluation_module.test_training(
            evaluation_functions,
            covariance_list,
            parameter_values,
            covariance_type,
        )

    return evaluation_functions


def generate_covariance(
    emulator_model_name,
    model_kind,
    covariance_list,
    parameter_values,
    **kwargs,
):

    emulator_covariance_dict = {}

    if model_kind in ["density", "full", "density_velocity"]:
        emulator_covariance_dict["gg"] = compute_cov(
            emulator_model_name,
            covariance_list,
            parameter_values,
            "gg",
            **kwargs,
        )

    if model_kind in ["velocity", "full", "density_velocity"]:
        emulator_covariance_dict["vv"] = compute_cov(
            emulator_model_name,
            covariance_list,
            parameter_values,
            "vv",
            **kwargs,
        )

    if model_kind == "full":
        emulator_covariance_dict["gv"] = compute_cov(
            emulator_model_name,
            covariance_list,
            parameter_values,
            "gv",
            **kwargs,
        )

    return emulator_covariance_dict
