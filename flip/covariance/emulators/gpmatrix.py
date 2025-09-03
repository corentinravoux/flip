import numpy as np

from flip.utils import create_log

log = create_log()

try:
    import GPy
except:
    log.add(
        "Install GPy to use the gpmatrix emulator",
        level="warning",
    )


_emulator_type = "matrix"

default_kernel_variance = GPy.kern.Exponential(1) + GPy.kern.Poly(1)
default_kernel_non_diagonal = GPy.kern.Exponential(1) + GPy.kern.Poly(1)


def train(
    square_covariance,
    output_variance,
    output_non_diagonal,
    parameter_values,
    kernel_variance=None,
    kernel_non_diagonal=None,
    verbose=False,
    num_restarts_variance=None,
    num_restarts_non_diagonal=None,
    **kwargs,
):

    if square_covariance:
        if kernel_variance is None:
            kernel_variance = default_kernel_variance
    if kernel_non_diagonal is None:
        kernel_non_diagonal = default_kernel_non_diagonal

    if square_covariance:
        gp_models_variance = []
    else:
        gp_models_variance = None

    gp_models_non_diagonal = []

    for j in range(len(output_non_diagonal)):

        if square_covariance:
            model_variance = GPy.models.GPRegression(
                parameter_values,
                output_variance[j][:, np.newaxis],
                kernel_variance,
            )
            model_variance.optimize(messages=verbose)
            if num_restarts_variance is not None:
                model_variance.optimize_restarts(num_restarts=num_restarts_variance)

            gp_models_variance.append(model_variance)

        model_non_diagonal = GPy.models.GPRegression(
            parameter_values, output_non_diagonal[j], kernel_non_diagonal
        )
        model_non_diagonal.optimize(messages=verbose)
        if num_restarts_non_diagonal is not None:
            model_non_diagonal.optimize_restarts(num_restarts=num_restarts_non_diagonal)

        gp_models_non_diagonal.append(model_non_diagonal)

    gp_evaluation_dictionary_variance = None
    gp_evaluation_dictionary_non_diagonal = None

    return (
        gp_models_variance,
        gp_models_non_diagonal,
        gp_evaluation_dictionary_variance,
        gp_evaluation_dictionary_non_diagonal,
    )


def evaluate(
    model,
    evaluation_value,
    evaluation_dictionary,
):
    output = model.predict(evaluation_value)
    return output
