import importlib

import numpy as np

import flip.covariance.emulators.gpmatrix as gpmatrix
import flip.covariance.emulators.nnmatrix as nnmatrix
from flip.covariance import cov_utils
from flip.utils import create_log

log = create_log()

_avail_emulator_models = [
    "gpmatrix",
    "nnmatrix",
]


def compute_cov(
    emulator_model_name,
    covariance_list,
    parameter_values,
    emulator_parameter_names,
    covariance_type,
    test_emulator=False,
    **kwargs,
):
    if len(covariance_list) != len(parameter_values):
        raise ValueError(
            "covariance_list and parameter_values must have the same length."
        )
    if emulator_model_name not in _avail_emulator_models:
        log.add(
            f"Model {emulator_model_name} not available."
            f"Please choose between: {_avail_emulator_models}"
        )

    emulator_module = importlib.import_module(
        f"flip.covariance.emulators.{emulator_model_name}"
    )

    emulator_type = emulator_module._emulator_type
    # Only implemented for matrix emulators, so far.

    square_covariance, emulator_output_variance, emulator_output_non_diagonal = (
        prepare_covariance_matrices(
            covariance_list,
            covariance_type,
        )
    )

    models = emulator_module.train(
        square_covariance,
        emulator_output_variance,
        emulator_output_non_diagonal,
        parameter_values,
        **kwargs,
    )

    evaluation_functions = return_evaluation_functions(
        models,
        emulator_module,
        emulator_parameter_names,
        covariance_type,
        **kwargs,
    )

    if test_emulator:
        parameter_values_dict = [
            {
                emulator_parameter_names[j]: parameter_values[i][j]
                for j in range(len(emulator_parameter_names))
            }
            for i in range(len(parameter_values))
        ]
        test_training(
            evaluation_functions,
            covariance_list,
            parameter_values_dict,
            covariance_type,
        )

    return evaluation_functions


def prepare_covariance_matrices(
    covariance_list,
    covariance_type,
):
    if covariance_type[0] == covariance_type[1]:
        square_covariance = True
    else:
        square_covariance = False

    emulator_output_non_diagonal = [
        [] for _ in range(len(covariance_list[0].covariance_dict[covariance_type]))
    ]
    if square_covariance:
        emulator_output_variance = [
            [] for _ in range(len(covariance_list[0].covariance_dict[covariance_type]))
        ]
    else:
        emulator_output_variance = None

    for i in range(len(covariance_list)):
        covariance_list[i].compute_flat_covariance(verbose=False)
        for j in range(len(covariance_list[i].covariance_dict[covariance_type])):

            if square_covariance:
                emulator_output_variance[j].append(
                    covariance_list[i].covariance_dict[covariance_type][j][0]
                )

                emulator_output_non_diagonal[j].append(
                    covariance_list[i].covariance_dict[covariance_type][j][1:]
                )
            else:
                emulator_output_non_diagonal[j].append(
                    covariance_list[i].covariance_dict[covariance_type][j]
                )

    return (
        square_covariance,
        np.array(emulator_output_variance),
        np.array(emulator_output_non_diagonal),
    )


def return_evaluation_functions(
    models,
    emulator_module,
    emulator_parameter_names,
    covariance_type,
    **kwargs,
):

    if covariance_type[0] == covariance_type[1]:
        square_covariance = True
        models_variance, models_non_diagonal = models[0], models[1]
        evaluation_dictionary_variance, evaluation_dictionary_non_diagonal = (
            models[2],
            models[3],
        )
    else:
        square_covariance = False
        models_non_diagonal = models[1]
        evaluation_dictionary_non_diagonal = models[3]

    evaluation_functions = []

    for j in range(len(models_non_diagonal)):
        if square_covariance:
            model_variance = models_variance[j]
        model_non_diagonal = models_non_diagonal[j]

        if square_covariance:

            def evaluate(
                parameter_values_dict,
                return_emulator_variance=False,
            ):

                parameter_value = [
                    parameter_values_dict[name] for name in emulator_parameter_names
                ]
                evaluation_value = np.array(parameter_value)[np.newaxis, :]
                output_non_diagonal = emulator_module.evaluate(
                    model_non_diagonal,
                    evaluation_value,
                    evaluation_dictionary_variance,
                )
                output_covariance = output_non_diagonal[0][0]
                output_diagonal = emulator_module.evaluate(
                    model_variance, evaluation_value
                )
                output_covariance = np.insert(
                    output_covariance, 0, output_diagonal[0][0]
                )
                output_covariance = cov_utils.return_matrix_covariance(
                    output_covariance
                )
                variance_emulator = [
                    output_diagonal[1][0, 0],
                    output_non_diagonal[1][0, 0],
                ]
                if return_emulator_variance:
                    return output_covariance, variance_emulator
                else:
                    return output_covariance

        else:

            def evaluate(
                parameter_value,
                return_emulator_variance=False,
            ):
                evaluation_value = np.array(parameter_value)[np.newaxis, np.newaxis]
                output_non_diagonal = emulator_module.evaluate(
                    model_non_diagonal,
                    evaluation_value,
                    evaluation_dictionary_non_diagonal,
                )
                output_covariance = output_non_diagonal[0][0]
                variance_emulator = output_non_diagonal[1][0, 0]

                if return_emulator_variance:
                    return output_covariance, variance_emulator
                else:
                    return output_covariance

        evaluation_functions.append(evaluate)

    return evaluation_functions


def test_training(
    evaluation_functions,
    covariance_list,
    parameter_values_dict,
    covariance_type,
    verbose=True,
    **kwargs,
):
    if covariance_type[0] == covariance_type[1]:
        square_covariance = True
    else:
        square_covariance = False
    errors = []
    for i in range(len(parameter_values_dict)):
        covariance_list[i].compute_matrix_covariance(verbose=False)
        errors.append([])
        for j in range(len(evaluation_functions)):
            evaluation = evaluation_functions[j](parameter_values_dict[i])
            covariance = covariance_list[i].covariance_dict[covariance_type][j]

            if square_covariance:
                error_diagonal = np.abs(evaluation[0][0] - covariance[0][0])
                relative_error_diagonal = np.abs(
                    evaluation[0][0] - covariance[0][0]
                ) / np.abs(covariance[0][0])
            else:
                error_diagonal = None
                relative_error_diagonal = None
            error_all = np.max(np.abs(evaluation - covariance))
            relative_error_all = np.max(
                np.abs(evaluation - covariance) / np.max(np.abs(covariance))
            )
            if verbose:
                log.add(
                    (
                        f"For parameter {i} and covariance type {covariance_type}\n"
                        f"Covariance term {j} gives: \n"
                        f"Maximal relative error all {relative_error_all}\n"
                    ),
                    level="info",
                )
                if square_covariance:
                    log.add(
                        f"Maximal relative error diagonal {relative_error_diagonal}, \n",
                        level="info",
                    )

            errors[i].append(
                [
                    error_diagonal,
                    relative_error_diagonal,
                    error_all,
                    relative_error_all,
                ]
            )

    return errors


def generate_covariance(
    emulator_model_name,
    model_kind,
    covariance_list,
    parameter_values,
    emulator_parameter_names,
    **kwargs,
):

    emulator_covariance_dict = {}

    if model_kind in ["density", "full", "density_velocity"]:
        emulator_covariance_dict["gg"] = compute_cov(
            emulator_model_name,
            covariance_list,
            parameter_values,
            emulator_parameter_names,
            "gg",
            **kwargs,
        )

    if model_kind in ["velocity", "full", "density_velocity"]:
        emulator_covariance_dict["vv"] = compute_cov(
            emulator_model_name,
            covariance_list,
            parameter_values,
            emulator_parameter_names,
            "vv",
            **kwargs,
        )

    if model_kind == "full":
        emulator_covariance_dict["gv"] = compute_cov(
            emulator_model_name,
            covariance_list,
            parameter_values,
            emulator_parameter_names,
            "gv",
            **kwargs,
        )

    return emulator_covariance_dict
