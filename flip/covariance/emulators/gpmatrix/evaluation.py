import numpy as np

from flip.utils import create_log

log = create_log()


def return_evaluation_functions(
    models,
    **kwargs,
):
    gp_models_variance, gp_models_non_diagonal = models[0], models[1]

    evaluation_functions = []

    for j in range(len(gp_models_variance)):
        model_variance = gp_models_variance[j]
        model_non_diagonal = gp_models_non_diagonal[j]

        def evaluate(
            parameter_value,
            return_emulator_variance=False,
        ):
            evaluation_value = np.array(parameter_value)[np.newaxis, np.newaxis]

            gp_output_diagonal = model_variance.predict(evaluation_value)
            gp_output_non_diagonal = model_non_diagonal.predict(evaluation_value)

            flat_covariance = gp_output_non_diagonal[0][0]
            flat_covariance = np.insert(flat_covariance, 0, gp_output_diagonal[0][0])

            variance_emulator = [
                gp_output_diagonal[1][0, 0],
                gp_output_non_diagonal[1][0, 0],
            ]

            if return_emulator_variance:
                return flat_covariance, variance_emulator
            else:
                return flat_covariance

        evaluation_functions.append(evaluate)

    return evaluation_functions


def test_training(
    evaluation_functions,
    covariance_list,
    parameter_values,
    covariance_type,
    **kwargs,
):

    errors = []
    for i in range(len(parameter_values)):
        covariance_list[i].compute_flat_covariance(verbose=False)
        errors.append([])
        for j in range(len(evaluation_functions)):
            evaluation = evaluation_functions[j](parameter_values[i])
            covariance = covariance_list[i].covariance_dict[covariance_type][j]

            error_diagonal = np.abs(evaluation[0] - covariance[0])
            relative_error_diagonal = np.abs(evaluation[0] - covariance[0]) / np.abs(
                covariance[0]
            )

            error_non_diagonal = np.max(np.abs(evaluation[1:] - covariance[1:]))
            relative_error_non_diagonal = np.max(
                np.abs(evaluation[1:] - covariance[1:]) / np.max(np.abs(covariance[1:]))
            )

            log.add(
                f"For parameter {i} and covariance type {covariance_type}\n"
                f"Covariance term {j} gives: \n"
                f"relative error diagonal {relative_error_diagonal}, \n"
                f"relative error non-diagonal {relative_error_non_diagonal}\n",
                level="info",
            )

            errors[i].append(
                [
                    error_diagonal,
                    relative_error_diagonal,
                    error_non_diagonal,
                    relative_error_non_diagonal,
                ]
            )

    return errors
