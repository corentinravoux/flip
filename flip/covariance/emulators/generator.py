import importlib

import numpy as np

from flip.covariance import cov_utils
from flip.covariance.emulators import prepare
from flip.utils import create_log

log = create_log()

_avail_emulator_models = [
    "gpmatrix",
    "nnmatrix",
    "skgpmatrix",
    "skgpscalar",
    "skgppowerspectrum",
]

_avail_emulator_types = [
    "matrix",
    "scalar",
    "power_spectrum",
]


def load_emulator_module(emulator_model_name):
    """Import an emulator backend module.

    Args:
        emulator_model_name: Backend name, one of ``_avail_emulator_models``.

    Returns:
        The backend module.

    Raises:
        ValueError: If the backend name is not available.
    """
    if emulator_model_name not in _avail_emulator_models:
        raise ValueError(
            f"Model {emulator_model_name} not available. "
            f"Please choose between: {_avail_emulator_models}"
        )
    return importlib.import_module(f"flip.covariance.emulators.{emulator_model_name}")


def compute_cov(
    emulator_model_name,
    covariance_list,
    parameter_values,
    emulator_parameter_names,
    covariance_type,
    test_emulator=False,
    power_spectrum_list=None,
    model_name=None,
    coordinates_density=None,
    coordinates_velocity=None,
    additional_parameters_values=None,
    los_definition="bisector",
    **kwargs,
):
    """Train an emulator for a given covariance type and return evaluators.

    The training data are prepared by :mod:`flip.covariance.emulators.prepare`
    according to the emulator type of the backend:

    - ``matrix``: from ``covariance_list``;
    - ``scalar`` and ``power_spectrum``: from ``power_spectrum_list`` and the
      flip model ``model_name`` (and the catalogue coordinates).

    Args:
        emulator_model_name: Backend name, one of ``_avail_emulator_models``.
        covariance_list: List of covariance objects with precomputed matrices
            (matrix emulators), None otherwise.
        parameter_values: Array ``(n_samples, n_params)`` training inputs.
        emulator_parameter_names: Names of parameters to pass at evaluation time.
        covariance_type: One of ``"gg"``, ``"vv"``, ``"gv"`` indicating block type.
        test_emulator: If True, runs a quick evaluation vs. training data and
            logs errors (matrix emulators only).
        power_spectrum_list: List of flip power spectrum dictionaries, one per
            parameter sample (scalar and power spectrum emulators).
        model_name: flip analytical model (scalar and power spectrum emulators).
        coordinates_density: Density coordinates (scalar and power spectrum emulators).
        coordinates_velocity: Velocity coordinates (scalar and power spectrum emulators).
        additional_parameters_values: Extra (fixed) parameters of the model kernels.
        los_definition: Line-of-sight convention.
        **kwargs: Extra keyword arguments forwarded to the preparation, to the
            backend ``train`` and to the evaluation functions.

    Returns:
        List of callables, one per covariance term, taking a parameter dict and returning a matrix.

    Raises:
        ValueError: If list lengths mismatch, model name is invalid, or inputs
            needed by the emulator type are missing.
    """
    emulator_module = load_emulator_module(emulator_model_name)
    emulator_type = emulator_module._emulator_type
    parameter_values = np.atleast_2d(np.asarray(parameter_values, dtype=float))
    if parameter_values.shape[0] == 1 and len(emulator_parameter_names) == 1:
        parameter_values = parameter_values.T
    if parameter_values.shape[1] != len(emulator_parameter_names):
        raise ValueError(
            "parameter_values must have shape (n_samples, n_params) with "
            "n_params = len(emulator_parameter_names)."
        )

    if emulator_type == "matrix":
        if covariance_list is None:
            raise ValueError("Matrix emulators need covariance_list.")
        if len(covariance_list) != len(parameter_values):
            raise ValueError(
                "covariance_list and parameter_values must have the same length."
            )
        square_covariance, emulator_output_variance, emulator_output_non_diagonal = (
            prepare.prepare_matrix(
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
            covariance_list[0].number_densities,
            covariance_list[0].number_velocities,
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

    if power_spectrum_list is None or model_name is None:
        raise ValueError(
            f"{emulator_type} emulators need power_spectrum_list and model_name."
        )
    if test_emulator:
        log.add(
            "test_emulator is only available for matrix emulators.",
            level="warning",
        )

    if emulator_type == "scalar":
        pair_geometry = prepare.compute_pair_geometry(
            covariance_type,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            los_definition=los_definition,
        )
        r_max = kwargs.pop("r_max", None)
        if r_max is None:
            r_max = 1.05 * np.max(pair_geometry[0])
        elif np.max(pair_geometry[0]) > r_max:
            log.add(
                f"Pair separations up to {np.max(pair_geometry[0]):.1f} exceed "
                f"r_max = {r_max:.1f}: the scalar emulator extrapolates.",
                level="warning",
            )
        (
            square_covariance,
            input_variance,
            output_variance,
            input_non_diagonal,
            output_non_diagonal,
            scalar_dictionary,
        ) = prepare.prepare_scalar(
            power_spectrum_list,
            parameter_values,
            model_name,
            covariance_type,
            r_max,
            additional_parameters_values=additional_parameters_values,
            **kwargs,
        )
        models = emulator_module.train(
            square_covariance,
            input_variance,
            output_variance,
            input_non_diagonal,
            output_non_diagonal,
            **kwargs,
        )
        return return_evaluation_functions_scalar(
            models,
            emulator_module,
            emulator_parameter_names,
            covariance_type,
            scalar_dictionary,
            pair_geometry,
            **kwargs,
        )

    if emulator_type == "power_spectrum":
        (
            square_covariance,
            coefficients,
            basis_covariances,
            spectrum_indices,
            _,
        ) = prepare.prepare_power_spectrum(
            power_spectrum_list,
            parameter_values,
            model_name,
            covariance_type,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=additional_parameters_values,
            los_definition=los_definition,
            **kwargs,
        )
        models = emulator_module.train(
            coefficients,
            parameter_values,
            **kwargs,
        )
        return return_evaluation_functions_power_spectrum(
            models,
            emulator_module,
            emulator_parameter_names,
            covariance_type,
            basis_covariances,
            spectrum_indices,
            **kwargs,
        )

    raise ValueError(
        f"Emulator type {emulator_type} not available. "
        f"Please choose between: {_avail_emulator_types}"
    )


def prepare_covariance_matrices(
    covariance_list,
    covariance_type,
):
    """Extract and stack covariance training targets for matrix emulators.

    Kept for backward compatibility, see
    :func:`flip.covariance.emulators.prepare.prepare_matrix`.
    """
    return prepare.prepare_matrix(covariance_list, covariance_type)


def _parameter_vector(parameter_values_dict, emulator_parameter_names):
    return np.array(
        [parameter_values_dict[name] for name in emulator_parameter_names],
        dtype=float,
    )


def return_evaluation_functions(
    models,
    emulator_module,
    emulator_parameter_names,
    covariance_type,
    number_densities=None,
    number_velocities=None,
    **kwargs,
):
    """Wrap matrix backend models into per-term evaluation functions.

    Each returned function accepts a parameter dict mapping names to values and
    returns a matrix. Each function is built by a factory, so it keeps the
    models of its own term.

    Args:
        models: Tuple of backend-trained models and evaluation dictionaries.
        emulator_module: Module providing ``evaluate(model, x, dict)`` function.
        emulator_parameter_names: Names used to extract parameter order from dict.
        covariance_type: Block type ``("gg"|"vv"|"gv")``.
        number_densities: (Float) number of density tracers used for the 'gv'
        number_velocities: (Float) number of velocity tracers used for the 'gv'
        **kwargs: Extra keyword arguments (unused).

    Returns:
        List of evaluation callables for each covariance term.
    """
    (
        models_variance,
        models_non_diagonal,
        evaluation_dictionary_variance,
        evaluation_dictionary_non_diagonal,
    ) = models

    def make_square_function(
        model_variance,
        model_non_diagonal,
        dictionary_variance,
        dictionary_non_diagonal,
    ):
        def evaluation_function(
            parameter_values_dict,
            return_emulator_variance=False,
        ):
            evaluation_value = _parameter_vector(
                parameter_values_dict, emulator_parameter_names
            )[np.newaxis, :]
            output_non_diagonal = emulator_module.evaluate(
                model_non_diagonal,
                evaluation_value,
                dictionary_non_diagonal,
            )
            output_diagonal = emulator_module.evaluate(
                model_variance,
                evaluation_value,
                dictionary_variance,
            )
            output_covariance = np.insert(
                np.ravel(output_non_diagonal[0]), 0, np.ravel(output_diagonal[0])[0]
            )
            output_covariance = cov_utils.return_matrix_covariance(output_covariance)
            if return_emulator_variance:
                variance_emulator = [output_diagonal[1], output_non_diagonal[1]]
                return output_covariance, variance_emulator
            return output_covariance

        return evaluation_function

    def make_cross_function(model_non_diagonal, dictionary_non_diagonal):
        def evaluation_function(
            parameter_values_dict,
            number_densities,
            number_velocities,
            return_emulator_variance=False,
        ):
            evaluation_value = _parameter_vector(
                parameter_values_dict, emulator_parameter_names
            )[np.newaxis, :]
            output_non_diagonal = emulator_module.evaluate(
                model_non_diagonal,
                evaluation_value,
                dictionary_non_diagonal,
            )
            output_covariance = cov_utils.return_matrix_covariance_cross(
                np.ravel(output_non_diagonal[0]), number_densities, number_velocities
            )
            if return_emulator_variance:
                return output_covariance, output_non_diagonal[1]
            return output_covariance

        return evaluation_function

    if prepare.is_square_covariance(covariance_type):
        return [
            make_square_function(
                models_variance[j],
                models_non_diagonal[j],
                evaluation_dictionary_variance[j],
                evaluation_dictionary_non_diagonal[j],
            )
            for j in range(len(models_non_diagonal))
        ]
    return [
        make_cross_function(
            models_non_diagonal[j],
            evaluation_dictionary_non_diagonal[j],
        )
        for j in range(len(models_non_diagonal))
    ]


def _warn_outside_training(parameter_vector, parameter_bounds, warned):
    if warned[0]:
        return
    if np.any(parameter_vector < parameter_bounds[:, 0]) or np.any(
        parameter_vector > parameter_bounds[:, 1]
    ):
        log.add(
            f"Emulator evaluated at {parameter_vector}, outside its training "
            f"bounds {parameter_bounds.tolist()}: it extrapolates. "
            "Warning given once per evaluation function.",
            level="warning",
        )
        warned[0] = True


def return_evaluation_functions_scalar(
    models,
    emulator_module,
    emulator_parameter_names,
    covariance_type,
    scalar_dictionary,
    pair_geometry,
    positive_semidefinite=True,
    size_chunk=10_000,
    **kwargs,
):
    """Wrap scalar backend models into per-term evaluation functions.

    The scalar models predict one covariance element per pair of the
    catalogue; the elements are then assembled into a matrix.

    Args:
        models: Tuple of backend-trained models and evaluation dictionaries.
        emulator_module: Module providing ``evaluate(model, x, dict)`` function.
        emulator_parameter_names: Names used to extract parameter order from dict.
        covariance_type: Block type ``("gg"|"vv"|"gv")``.
        scalar_dictionary: Parameter bounds and ``r_max`` of the training set.
        pair_geometry: Array ``(3, n_pairs)`` with ``r, theta, phi`` of the catalogue.
        positive_semidefinite: Clip the negative eigenvalues of the emulated
            square blocks to zero. Element errors of the scalar emulators leave
            negative eigenvalues, which can make the total covariance fail the
            Cholesky decomposition at small noise.
        size_chunk: Number of pairs predicted at once.
        **kwargs: Extra keyword arguments (unused).

    Returns:
        List of evaluation callables for each covariance term.
    """
    (
        models_variance,
        models_non_diagonal,
        evaluation_dictionary_variance,
        evaluation_dictionary_non_diagonal,
    ) = models
    parameter_bounds = scalar_dictionary["parameter_bounds"]
    r_max = scalar_dictionary["r_max"]
    square_covariance = prepare.is_square_covariance(covariance_type)
    r, theta, phi = pair_geometry

    def predict_non_diagonal(model, dictionary, parameter_vector):
        output = []
        for i in range(0, len(r), size_chunk):
            features = prepare.scalar_features(
                parameter_vector,
                r[i : i + size_chunk],
                theta[i : i + size_chunk],
                phi[i : i + size_chunk],
                parameter_bounds,
                r_max,
            )
            output.append(
                np.ravel(emulator_module.evaluate(model, features, dictionary)[0])
            )
        return np.concatenate(output)

    def make_function(j):
        warned = [False]

        def evaluation_function(
            parameter_values_dict,
            number_densities=None,
            number_velocities=None,
        ):
            parameter_vector = _parameter_vector(
                parameter_values_dict, emulator_parameter_names
            )
            _warn_outside_training(parameter_vector, parameter_bounds, warned)
            output_non_diagonal = predict_non_diagonal(
                models_non_diagonal[j],
                evaluation_dictionary_non_diagonal[j],
                parameter_vector,
            )
            if not square_covariance:
                return cov_utils.return_matrix_covariance_cross(
                    output_non_diagonal, number_densities, number_velocities
                )
            output_diagonal = emulator_module.evaluate(
                models_variance[j],
                prepare.scale_parameters(parameter_vector, parameter_bounds),
                evaluation_dictionary_variance[j],
            )
            output_covariance = cov_utils.return_matrix_covariance(
                np.insert(output_non_diagonal, 0, np.ravel(output_diagonal[0])[0])
            )
            if positive_semidefinite:
                eigenvalues, eigenvectors = np.linalg.eigh(output_covariance)
                output_covariance = (
                    eigenvectors * np.clip(eigenvalues, 0, None)
                ) @ eigenvectors.T
            return output_covariance

        return evaluation_function

    return [make_function(j) for j in range(len(models_non_diagonal))]


def return_evaluation_functions_power_spectrum(
    models,
    emulator_module,
    emulator_parameter_names,
    covariance_type,
    basis_covariances,
    spectrum_indices,
    **kwargs,
):
    """Wrap power spectrum backend models into per-term evaluation functions.

    The covariance of term ``t`` is ``sum_n a_sn(theta) C_t[b_sn]``, with the
    coefficients ``a_sn`` predicted by the backend for the spectrum ``s`` of
    term ``t``.

    Args:
        models: Tuple ``(models, evaluation_dictionaries)`` of the backend,
            indexed by spectrum then basis component.
        emulator_module: Module providing ``evaluate(model, x, dict)`` function.
        emulator_parameter_names: Names used to extract parameter order from dict.
        covariance_type: Block type ``("gg"|"vv"|"gv")``.
        basis_covariances: Array ``(n_terms, n_basis, n_flat)`` of flat covariances.
        spectrum_indices: Spectrum index of each term.
        **kwargs: Extra keyword arguments (unused).

    Returns:
        List of evaluation callables for each covariance term.
    """
    coefficient_models, evaluation_dictionaries = models
    square_covariance = prepare.is_square_covariance(covariance_type)

    def make_function(j):
        spectrum_index = spectrum_indices[j]
        warned = [False]

        def evaluation_function(
            parameter_values_dict,
            number_densities=None,
            number_velocities=None,
        ):
            parameter_vector = _parameter_vector(
                parameter_values_dict, emulator_parameter_names
            )
            _warn_outside_training(
                parameter_vector,
                evaluation_dictionaries[spectrum_index][0]["parameter_bounds"],
                warned,
            )
            coefficients = np.array(
                [
                    np.ravel(
                        emulator_module.evaluate(
                            model,
                            parameter_vector[np.newaxis, :],
                            evaluation_dictionary,
                        )[0]
                    )[0]
                    for model, evaluation_dictionary in zip(
                        coefficient_models[spectrum_index],
                        evaluation_dictionaries[spectrum_index],
                    )
                ]
            )
            flat_covariance = coefficients @ basis_covariances[j]
            if square_covariance:
                return cov_utils.return_matrix_covariance(flat_covariance)
            return cov_utils.return_matrix_covariance_cross(
                flat_covariance, number_densities, number_velocities
            )

        return evaluation_function

    return [make_function(j) for j in range(len(basis_covariances))]


def test_training(
    evaluation_functions,
    covariance_list,
    parameter_values_dict,
    covariance_type,
    verbose=True,
    **kwargs,
):
    """Compute emulator errors against training covariances for sanity checks.

    Args:
        evaluation_functions: List of term evaluators returned by ``return_evaluation_functions``.
        covariance_list: List of covariance objects (ground truth).
        parameter_values_dict: List of parameter dicts matching training inputs.
        covariance_type: Block type ``("gg"|"vv"|"gv")``.
        verbose: If True, logs per-term maximal relative errors.
        **kwargs: Extra keyword arguments (unused).

    Returns:
        Nested list of errors per sample and term: ``[ [err_diag, rel_err_diag, err_all, rel_err_all], ... ]``.
    """
    square_covariance = prepare.is_square_covariance(covariance_type)
    errors = []
    for i in range(len(parameter_values_dict)):
        errors.append([])
        for j in range(len(evaluation_functions)):
            covariance = np.asarray(
                covariance_list[i].covariance_dict[covariance_type][j]
            )
            if not covariance_list[i].matrix_form:
                if square_covariance:
                    covariance = cov_utils.return_matrix_covariance(covariance)
                else:
                    covariance = cov_utils.return_matrix_covariance_cross(
                        covariance,
                        covariance_list[i].number_densities,
                        covariance_list[i].number_velocities,
                    )

            if square_covariance:
                evaluation = evaluation_functions[j](parameter_values_dict[i])
                error_diagonal = np.abs(evaluation[0][0] - covariance[0][0])
                relative_error_diagonal = np.abs(
                    evaluation[0][0] - covariance[0][0]
                ) / np.abs(covariance[0][0])
            else:
                evaluation = evaluation_functions[j](
                    parameter_values_dict[i],
                    covariance_list[0].number_densities,
                    covariance_list[0].number_velocities,
                )
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
    """Generate emulator evaluators for requested covariance blocks.

    Args:
        emulator_model_name: Backend name.
        model_kind: One of ``"density"``, ``"velocity"``, ``"full"``, ``"density_velocity"``.
        covariance_list: List of covariance objects for training (matrix
            emulators), None otherwise.
        parameter_values: Array ``(n_samples, n_params)`` training inputs.
        emulator_parameter_names: Parameter names used at evaluation.
        **kwargs: Forwarded to :func:`compute_cov`.

    Returns:
        Dict with keys among ``{"gg","vv","gv"}`` mapping to lists of evaluator functions.
    """

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
