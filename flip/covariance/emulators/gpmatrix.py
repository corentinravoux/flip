import numpy as np

from flip.utils import create_log

log = create_log()

try:
    import GPy

    gpy_installed = True
except ImportError:
    gpy_installed = False
    log.add(
        "Install GPy to use the gpmatrix emulator",
        level="warning",
    )


_emulator_type = "matrix"

if gpy_installed:
    default_kernel_variance = GPy.kern.Exponential(1) + GPy.kern.Poly(1)
    default_kernel_non_diagonal = GPy.kern.Exponential(1) + GPy.kern.Poly(1)
else:
    default_kernel_variance = None
    default_kernel_non_diagonal = None


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
    """Train Gaussian Process emulators for covariance matrices.

    Trains one GP per covariance term. For square covariances (gg or vv), it
    trains both a GP for the diagonal element (variance) and one for the
    non-diagonal elements. For rectangular covariances (gv), only the
    non-diagonal GP is trained.

    Args:
        square_covariance: Whether the covariance is square (gg or vv) or not (gv).
        output_variance: Array of shape ``(n_terms, n_samples)`` containing diagonal values per sample. Ignored if not square.
        output_non_diagonal: Array of shape ``(n_terms, n_samples, n_nd)`` containing flattened non-diagonal values per sample.
        parameter_values: Array of shape ``(n_samples, n_params)`` with emulator input parameters.
        kernel_variance: Optional GPy kernel to use for variance models. Defaults to ``Exponential + Poly`` if GPy is available.
        kernel_non_diagonal: Optional GPy kernel to use for non-diagonal models. Defaults to ``Exponential + Poly`` if GPy is available.
        verbose: If True, prints optimization messages.
        num_restarts_variance: Optional number of optimizer restarts for variance GPs.
        num_restarts_non_diagonal: Optional number of optimizer restarts for non-diagonal GPs.
        **kwargs: Extra keyword arguments (unused).

    Returns:
        Tuple containing:
        - ``gp_models_variance``: List of variance GP models or ``None``.
        - ``gp_models_non_diagonal``: List of non-diagonal GP models.
        - ``gp_evaluation_dictionary_variance``: Per-term evaluation dictionaries or ``None``.
        - ``gp_evaluation_dictionary_non_diagonal``: Per-term evaluation dictionaries.

    Raises:
        ImportError: If GPy is not installed when training is requested.
    """

    if square_covariance:
        if kernel_variance is None:
            kernel_variance = default_kernel_variance
        gp_models_variance = []
        gp_evaluation_dictionary_variance = [None for j in range(len(output_variance))]
    else:
        gp_models_variance = None
        gp_evaluation_dictionary_variance = None
    if kernel_non_diagonal is None:
        kernel_non_diagonal = default_kernel_non_diagonal

    gp_models_non_diagonal = []
    gp_evaluation_dictionary_non_diagonal = [
        None for j in range(len(output_non_diagonal))
    ]

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
    """Evaluate a trained GP model at given parameter values.

    Args:
        model: Trained GPy GPRegression model.
        evaluation_value: Array of shape ``(1, n_params)`` with input parameters.
        evaluation_dictionary: Placeholder for API consistency (unused for GPy).

    Returns:
        Tuple ``(mean, std)`` as returned by ``GPy.models.GPRegression.predict``.
    """
    output = model.predict(evaluation_value)
    return output
