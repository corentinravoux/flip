import numpy as np

try:
    from sklearn.base import clone
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    sklearngp_installed = True

except ImportError:
    sklearngp_installed = False


_emulator_type = "power_spectrum"


def default_kernel(number_parameters):
    """Kernel of the coefficient GPs: inputs are the parameters scaled to [-1, 1]."""
    return ConstantKernel(1.0) * RBF(
        np.ones(number_parameters), (1e-2, 1e3)
    ) + WhiteKernel(1e-10, (1e-14, 1e-4))


def _scale(parameter_values, parameter_bounds):
    low, high = parameter_bounds[:, 0], parameter_bounds[:, 1]
    return 2 * (np.atleast_2d(parameter_values) - low) / (high - low) - 1


def train(
    coefficients,
    parameter_values,
    kernel=None,
    num_restarts=2,
    random_state=0,
    **kwargs,
):
    """Train scikit-learn GP emulators of power spectrum basis coefficients.

    One GP per input spectrum and per basis component. A single multi-output
    GP shares one length scale between coefficients of very different shapes;
    its optimization collapsed in tests, hence one GP per coefficient.

    Args:
        coefficients: Array ``(n_spectra, n_samples, n_basis)`` of basis coefficients.
        parameter_values: Array ``(n_samples, n_params)`` emulator inputs.
        kernel: Kernel of the coefficient GPs; defaults to ``Constant * RBF + White``.
        num_restarts: Optimizer restarts per GP.
        random_state: Seed of the optimizer restarts.
        **kwargs: Extra keyword arguments (unused).

    Returns:
        Tuple ``(gp_models, gp_evaluation_dictionaries)``, both nested lists
        indexed by spectrum then basis component.
    """
    if not sklearngp_installed:
        raise ImportError("Install scikit-learn to use the skgppowerspectrum emulator")

    parameter_values = np.atleast_2d(parameter_values)
    parameter_bounds = np.column_stack(
        [parameter_values.min(axis=0), parameter_values.max(axis=0)]
    )
    if kernel is None:
        kernel = default_kernel(parameter_values.shape[1])
    scaled_parameter_values = _scale(parameter_values, parameter_bounds)

    gp_models, gp_evaluation_dictionaries = [], []
    for coefficients_spectrum in coefficients:
        gp_models_spectrum = []
        for n in range(coefficients_spectrum.shape[1]):
            model = GaussianProcessRegressor(
                kernel=clone(kernel),
                normalize_y=True,
                n_restarts_optimizer=num_restarts,
                random_state=random_state,
            )
            model.fit(scaled_parameter_values, coefficients_spectrum[:, n])
            gp_models_spectrum.append(model)
        gp_models.append(gp_models_spectrum)
        gp_evaluation_dictionaries.append(
            [{"parameter_bounds": parameter_bounds} for _ in gp_models_spectrum]
        )

    return gp_models, gp_evaluation_dictionaries


def evaluate(
    model,
    evaluation_value,
    evaluation_dictionary,
):
    """Evaluate a scikit-learn GP model of one basis coefficient.

    Args:
        model: Trained ``GaussianProcessRegressor``.
        evaluation_value: Array ``(1, n_params)`` input values.
        evaluation_dictionary: Dict with the training ``parameter_bounds``.

    Returns:
        Tuple ``(mean, std)`` from ``GaussianProcessRegressor.predict``.
    """
    return model.predict(
        _scale(evaluation_value, evaluation_dictionary["parameter_bounds"]),
        return_std=True,
    )
