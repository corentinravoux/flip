import numpy as np

try:
    from sklearn.base import clone
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    sklearngp_installed = True

except ImportError:
    sklearngp_installed = False


_emulator_type = "scalar"


def default_kernel_variance(number_parameters):
    """Kernel of the variance GPs: inputs are the parameters scaled to [-1, 1]."""
    return ConstantKernel(1.0) * RBF(np.ones(number_parameters), (1e-3, 1e4))


def default_kernel_non_diagonal(number_features):
    """Kernel of the non-diagonal GPs, with one length scale per feature (ARD).

    Features are the scaled parameters, ``r / r_max`` and six angular
    functions (see :func:`flip.covariance.emulators.prepare.scalar_features`).
    The length scale of ``r / r_max`` starts at 0.1, the others at 1. Features
    that do not matter end with a large length scale.
    """
    number_parameters = number_features - 7
    length_scale = np.concatenate([np.ones(number_parameters), [0.1], np.ones(6)])
    return ConstantKernel(1.0) * RBF(length_scale, (1e-3, 1e4)) + WhiteKernel(
        1e-6, (1e-12, 1e-2)
    )


def train(
    square_covariance,
    input_variance,
    output_variance,
    input_non_diagonal,
    output_non_diagonal,
    kernel_variance=None,
    kernel_non_diagonal=None,
    num_restarts_variance=0,
    num_restarts_non_diagonal=2,
    random_state=0,
    **kwargs,
):
    """Train scikit-learn GP emulators of single covariance elements.

    One GP per covariance term predicts a non-diagonal element from the
    parameters and the pair geometry. For square covariances (gg/vv), a second
    GP per term predicts the variance from the parameters.

    Training scales as ``O(n_points^3)`` and evaluation as
    ``O(n_points x n_pairs)``: keep ``n_points`` to a few thousand.

    Args:
        square_covariance: Whether covariance is square (gg/vv) or rectangular (gv).
        input_variance: Array ``(n_samples, n_params)`` of scaled parameters; ignored if not square.
        output_variance: Array ``(n_terms, n_samples)`` of variances; ignored if not square.
        input_non_diagonal: Array ``(n_points, n_features)`` of features.
        output_non_diagonal: Array ``(n_terms, n_points)`` of covariance elements.
        kernel_variance: Kernel for variance GP; defaults to ``Constant * RBF``.
        kernel_non_diagonal: Kernel for non-diagonal GP; defaults to
            ``Constant * RBF(ARD) + White``.
        num_restarts_variance: Optimizer restarts for variance GP.
        num_restarts_non_diagonal: Optimizer restarts for non-diagonal GP.
        random_state: Seed of the optimizer restarts.
        **kwargs: Extra keyword arguments (unused).

    Returns:
        Tuple of models and evaluation dictionaries analogous to other emulator backends.
    """
    if not sklearngp_installed:
        raise ImportError("Install scikit-learn to use the skgpscalar emulator")

    if kernel_non_diagonal is None:
        kernel_non_diagonal = default_kernel_non_diagonal(input_non_diagonal.shape[1])

    gp_models_non_diagonal = []
    for output in output_non_diagonal:
        model_non_diagonal = GaussianProcessRegressor(
            kernel=clone(kernel_non_diagonal),
            normalize_y=True,
            n_restarts_optimizer=num_restarts_non_diagonal,
            random_state=random_state,
        )
        model_non_diagonal.fit(input_non_diagonal, output)
        gp_models_non_diagonal.append(model_non_diagonal)
    gp_evaluation_dictionary_non_diagonal = [None for _ in output_non_diagonal]

    if square_covariance:
        if kernel_variance is None:
            kernel_variance = default_kernel_variance(input_variance.shape[1])
        gp_models_variance = []
        for output in output_variance:
            model_variance = GaussianProcessRegressor(
                kernel=clone(kernel_variance),
                normalize_y=True,
                n_restarts_optimizer=num_restarts_variance,
                random_state=random_state,
            )
            model_variance.fit(input_variance, output)
            gp_models_variance.append(model_variance)
        gp_evaluation_dictionary_variance = [None for _ in output_variance]
    else:
        gp_models_variance = None
        gp_evaluation_dictionary_variance = None

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
    """Evaluate a scikit-learn GP model of covariance elements.

    The predictive standard deviation is not computed: for the non-diagonal
    model, it would cost ``O(n_points^2)`` per pair.

    Args:
        model: Trained ``GaussianProcessRegressor``.
        evaluation_value: Array ``(n, n_features)`` of input features.
        evaluation_dictionary: Placeholder for API consistency (unused).

    Returns:
        Tuple ``(mean, None)``.
    """
    return model.predict(evaluation_value), None
