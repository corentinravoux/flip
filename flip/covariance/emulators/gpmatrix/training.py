import GPy
import numpy as np

from flip.utils import create_log

log = create_log()

default_kernel_variance = GPy.kern.Exponential(1) + GPy.kern.Poly(1)
default_kernel_non_diagonal = GPy.kern.Exponential(1) + GPy.kern.Poly(1)


def train_model(
    covariance_list,
    parameter_values,
    covariance_type,
    kernel_variance=None,
    kernel_non_diagonal=None,
    verbose=False,
    num_restarts_variance=None,
    num_restarts_non_diagonal=None,
    **kwargs,
):

    if len(covariance_list) != len(parameter_values):
        raise ValueError("covariance_list and sigmau_list must have the same length.")

    input_data = parameter_values[:, np.newaxis]

    gp_output_non_diagonal = [
        [] for _ in range(len(covariance_list[0].covariance_dict[covariance_type]))
    ]

    gp_output_variance = [
        [] for _ in range(len(covariance_list[0].covariance_dict[covariance_type]))
    ]

    for i in range(len(covariance_list)):
        covariance_list[i].compute_flat_covariance(verbose=False)
        for j in range(len(covariance_list[i].covariance_dict[covariance_type])):

            gp_output_variance[j].append(
                covariance_list[i].covariance_dict[covariance_type][j][0]
            )
            gp_output_non_diagonal[j].append(
                covariance_list[i].covariance_dict[covariance_type][j][1:]
            )

    if kernel_variance is None:
        kernel_variance = default_kernel_variance
    if kernel_non_diagonal is None:
        kernel_non_diagonal = default_kernel_non_diagonal

    gp_models_variance = []
    gp_models_non_diagonal = []

    for j in range(len(gp_output_variance)):

        model_variance = GPy.models.GPRegression(
            input_data, np.array(gp_output_variance[j])[:, np.newaxis], kernel_variance
        )
        model_variance.optimize(messages=verbose)
        if num_restarts_variance is not None:
            model_variance.optimize_restarts(num_restarts=num_restarts_variance)

        gp_models_variance.append(model_variance)

        model_non_diagonal = GPy.models.GPRegression(
            input_data, np.array(gp_output_non_diagonal[j]), kernel_non_diagonal
        )
        model_non_diagonal.optimize(messages=verbose)
        if num_restarts_non_diagonal is not None:
            model_non_diagonal.optimize_restarts(num_restarts=num_restarts_non_diagonal)

        gp_models_non_diagonal.append(model_non_diagonal)

    return gp_models_variance, gp_models_non_diagonal
