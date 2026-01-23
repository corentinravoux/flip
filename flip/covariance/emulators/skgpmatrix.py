import numpy as np

from flip.utils import create_log

log = create_log()

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    sklearngp_installed = True

except ImportError:
    sklearngp_installed = False
    log.add(
        "Install scikit-learn to use the sklearnmatrix emulator",
        level="warning",
    )


_emulator_type = "matrix"

if sklearngp_installed:
    default_kernel_variance = RBF(length_scale=1.0) + ConstantKernel(constant_value=1.0)
    default_kernel_non_diagonal = RBF(length_scale=1.0) + ConstantKernel(
        constant_value=1.0
    )
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
    num_restarts_variance=0,
    num_restarts_non_diagonal=0,
    **kwargs,
):
    """Train scikit-learn GP emulators for covariance matrices.

    Args:
        square_covariance: Whether covariance is square (gg/vv) or rectangular (gv).
        output_variance: Array ``(n_terms, n_samples)`` of diagonal entries; ignored if not square.
        output_non_diagonal: Array ``(n_terms, n_samples, n_nd)`` of flattened non-diagonals.
        parameter_values: Array ``(n_samples, n_params)`` emulator inputs.
        kernel_variance: Kernel for variance GP; defaults to ``RBF + Constant``.
        kernel_non_diagonal: Kernel for non-diagonal GP; defaults to ``RBF + Constant``.
        num_restarts_variance: Optimizer restarts for variance GP.
        num_restarts_non_diagonal: Optimizer restarts for non-diagonal GP.
        **kwargs: Extra keyword arguments (unused).

    Returns:
        Tuple of models and evaluation dictionaries analogous to other emulator backends.
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

            model_variance = GaussianProcessRegressor(
                kernel=kernel_variance,
                n_restarts_optimizer=num_restarts_variance,
            )
            model_variance.fit(
                parameter_values,
                output_variance[j][:, np.newaxis],
            )
            gp_models_variance.append(model_variance)

        model_non_diagonal = GaussianProcessRegressor(
            kernel=kernel_non_diagonal,
            n_restarts_optimizer=num_restarts_non_diagonal,
        )
        model_non_diagonal.fit(
            parameter_values,
            output_non_diagonal[j],
        )
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
    """Evaluate a scikit-learn GP model.

    Args:
        model: Trained ``GaussianProcessRegressor``.
        evaluation_value: Array ``(1, n_params)`` input values.
        evaluation_dictionary: Placeholder for API consistency (unused).

    Returns:
        Tuple ``(mean, std)`` from ``GaussianProcessRegressor.predict``.
    """
    output = model.predict(evaluation_value, return_std=True)
    return output
