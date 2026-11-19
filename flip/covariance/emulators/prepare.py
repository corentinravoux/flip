"""Preparation of the training data of the covariance emulators.

Each emulator type needs its own training data:

- ``matrix``: flattened covariance blocks of a list of precomputed covariances.
- ``scalar``: single covariance elements, computed with the flip model for a set
  of parameter values and pair geometries ``(r, theta, phi)``.
- ``power_spectrum``: a reduced basis of the input power spectra, the flip
  covariance of each basis spectrum, and the coefficients of the training
  spectra on this basis.

The emulator backends only see arrays; everything that needs the flip model
or the catalogue happens here.
"""

import importlib

import numpy as np
from scipy.stats.qmc import LatinHypercube

from flip.covariance import cov_utils
from flip.covariance import generator as flip_generator
from flip.utils import create_log

log = create_log()


def _latin_hypercube(dimension, number_points, seed):
    """Latin hypercube sample, for scipy versions before and after SPEC 7."""
    try:
        sampler = LatinHypercube(d=dimension, rng=seed)
    except TypeError:
        sampler = LatinHypercube(d=dimension, seed=seed)
    return sampler.random(number_points)


def is_square_covariance(covariance_type):
    """Return True for auto-covariance blocks (``gg``, ``vv``)."""
    return covariance_type[0] == covariance_type[1]


def prepare_matrix(
    covariance_list,
    covariance_type,
):
    """Extract and stack covariance training targets for matrix emulators.

    The covariances of ``covariance_list`` are not modified: matrix-form
    covariances are flattened on a copy.

    Args:
        covariance_list: List of covariance objects.
        covariance_type: Block type ``("gg"|"vv"|"gv")``.

    Returns:
        Tuple ``(square_covariance, output_variance, output_non_diagonal)``.
        ``output_variance`` has shape ``(n_terms, n_samples)`` (None for ``gv``),
        ``output_non_diagonal`` has shape ``(n_terms, n_samples, n_flat)``.
    """
    square_covariance = is_square_covariance(covariance_type)

    number_terms = len(covariance_list[0].covariance_dict[covariance_type])
    emulator_output_non_diagonal = [[] for _ in range(number_terms)]
    emulator_output_variance = [[] for _ in range(number_terms)]

    for covariance in covariance_list:
        for j, covariance_term in enumerate(
            covariance.covariance_dict[covariance_type]
        ):
            covariance_term = np.asarray(covariance_term)
            if covariance.matrix_form:
                if square_covariance:
                    covariance_term = cov_utils.return_flat_covariance(covariance_term)
                else:
                    covariance_term = cov_utils.return_flat_cross_cov(covariance_term)

            if square_covariance:
                emulator_output_variance[j].append(covariance_term[0])
                emulator_output_non_diagonal[j].append(covariance_term[1:])
            else:
                emulator_output_non_diagonal[j].append(covariance_term)

    return (
        square_covariance,
        np.array(emulator_output_variance) if square_covariance else None,
        np.array(emulator_output_non_diagonal),
    )


def compute_pair_geometry(
    covariance_type,
    coordinates_density=None,
    coordinates_velocity=None,
    size_batch=10_000,
    los_definition="bisector",
):
    """Pair separations and angles of a catalogue, in flip's flat order.

    Args:
        covariance_type: Block type ``("gg"|"vv"|"gv")``.
        coordinates_density: Density coordinates ``(ra, dec, r_comoving)``.
        coordinates_velocity: Velocity coordinates ``(ra, dec, r_comoving)``.
        size_batch: Batch size of the pair computation.
        los_definition: Line-of-sight convention.

    Returns:
        Array ``(3, n_pairs)`` with ``r, theta, phi``.
    """
    return np.concatenate(
        flip_generator.compute_coordinates(
            covariance_type,
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            size_batch=size_batch,
            los_definition=los_definition,
        ),
        axis=1,
    )


def scale_parameters(parameter_values, parameter_bounds):
    """Map parameters from their training bounds to ``[-1, 1]``.

    Args:
        parameter_values: Array ``(n, n_params)``.
        parameter_bounds: Array ``(n_params, 2)`` of lower and upper bounds.

    Returns:
        Scaled array ``(n, n_params)``.
    """
    low, high = parameter_bounds[:, 0], parameter_bounds[:, 1]
    return 2 * (np.atleast_2d(parameter_values) - low) / (high - low) - 1


def scalar_features(
    parameter_values,
    r,
    theta,
    phi,
    parameter_bounds,
    r_max,
):
    """Input features of the scalar emulators.

    Features are the scaled parameters, ``r / r_max`` and the cosine and sine
    of ``theta``, ``phi`` and ``2 phi``.

    Args:
        parameter_values: Array ``(n_params,)`` shared by all pairs, or
            ``(n_pairs, n_params)``.
        r: Pair separations ``(n_pairs,)``.
        theta: Angles between the lines of sight ``(n_pairs,)``.
        phi: Angles between separation and line of sight ``(n_pairs,)``.
        parameter_bounds: Array ``(n_params, 2)`` of training bounds.
        r_max: Maximal separation of the training set.

    Returns:
        Array ``(n_pairs, n_params + 7)``.
    """
    scaled_parameters = np.broadcast_to(
        scale_parameters(parameter_values, parameter_bounds),
        (len(r), parameter_bounds.shape[0]),
    )
    return np.column_stack(
        [
            scaled_parameters,
            r / r_max,
            np.cos(theta),
            np.sin(theta),
            np.cos(phi),
            np.sin(phi),
            np.cos(2 * phi),
            np.sin(2 * phi),
        ]
    )


def prepare_scalar(
    power_spectrum_list,
    parameter_values,
    model_name,
    covariance_type,
    r_max,
    additional_parameters_values=None,
    number_training_points=1500,
    geometry_seed=0,
    hankel=True,
    kmin=None,
    **kwargs,
):
    """Compute single covariance elements for the scalar emulators.

    For each parameter sample, the flip model is evaluated on its own set of
    pair geometries, drawn from a Latin hypercube over
    ``r in [0, r_max]``, ``theta in [0, pi]``, ``phi in [0, pi]``. The geometries
    of the different samples do not repeat, so the training set is not a grid.

    Args:
        power_spectrum_list: List of flip power spectrum dictionaries, one per
            parameter sample.
        parameter_values: Array ``(n_samples, n_params)``.
        model_name: flip analytical model name.
        covariance_type: Block type ``("gg"|"vv"|"gv")``.
        r_max: Maximal separation of the training geometries.
        additional_parameters_values: Extra parameters of the model kernels.
        number_training_points: Approximate total number of training points
            per covariance term.
        geometry_seed: Seed of the Latin hypercube.
        hankel: Use the Hankel transform (flip default).
        kmin: Low-k regularisation of flip.
        **kwargs: Unused.

    Returns:
        Tuple ``(square_covariance, input_variance, output_variance,
        input_non_diagonal, output_non_diagonal, scalar_dictionary)``.
        Inputs are feature arrays, outputs have shape ``(n_terms, n_points)``.
        ``scalar_dictionary`` stores the parameter bounds and ``r_max``.
    """
    if len(power_spectrum_list) != len(parameter_values):
        raise ValueError(
            "power_spectrum_list and parameter_values must have the same length."
        )

    parameter_values = np.atleast_2d(np.asarray(parameter_values, dtype=float))
    square_covariance = is_square_covariance(covariance_type)
    number_samples = parameter_values.shape[0]
    number_geometry = int(np.ceil(number_training_points / number_samples))

    parameter_bounds = np.column_stack(
        [parameter_values.min(axis=0), parameter_values.max(axis=0)]
    )
    if np.any(parameter_bounds[:, 0] == parameter_bounds[:, 1]):
        raise ValueError("Each emulated parameter must take at least two values.")

    sample = _latin_hypercube(3, number_samples * number_geometry, geometry_seed)
    geometry = np.array(
        [sample[:, 0] * r_max, sample[:, 1] * np.pi, sample[:, 2] * np.pi]
    )

    output_variance, output_non_diagonal, input_parameters = [], [], []
    for i in range(number_samples):
        geometry_i = geometry[:, i * number_geometry : (i + 1) * number_geometry]
        coefficients = flip_generator.compute_coeficient(
            [geometry_i],
            model_name,
            covariance_type,
            power_spectrum_list[i][covariance_type],
            additional_parameters_values=additional_parameters_values,
            number_worker=1,
            hankel=hankel,
            kmin=kmin,
        )
        if square_covariance:
            output_variance.append(coefficients[:, 0])
            coefficients = coefficients[:, 1:]
        output_non_diagonal.append(coefficients)
        input_parameters.append(
            np.repeat(parameter_values[i][np.newaxis, :], number_geometry, axis=0)
        )

    input_non_diagonal = scalar_features(
        np.concatenate(input_parameters),
        geometry[0],
        geometry[1],
        geometry[2],
        parameter_bounds,
        r_max,
    )
    output_non_diagonal = np.concatenate(output_non_diagonal, axis=1)

    if square_covariance:
        input_variance = scale_parameters(parameter_values, parameter_bounds)
        output_variance = np.array(output_variance).T
    else:
        input_variance, output_variance = None, None

    scalar_dictionary = {
        "parameter_bounds": parameter_bounds,
        "r_max": r_max,
    }
    return (
        square_covariance,
        input_variance,
        output_variance,
        input_non_diagonal,
        output_non_diagonal,
        scalar_dictionary,
    )


def term_spectrum_indices(model_name, covariance_type):
    """Index of the input power spectrum used by each covariance term.

    Args:
        model_name: flip analytical model name.
        covariance_type: Block type ``("gg"|"vv"|"gv")``.

    Returns:
        List of integers, one per covariance term.
    """
    flip_terms = importlib.import_module(
        f"flip.covariance.analytical.{model_name}.flip_terms"
    )
    term_index_list = flip_terms.dictionary_terms[covariance_type]
    if flip_terms.multi_index_model:
        return [int(index[0]) for index in term_index_list]
    return list(range(len(term_index_list)))


def prepare_power_spectrum(
    power_spectrum_list,
    parameter_values,
    model_name,
    covariance_type,
    coordinates_density=None,
    coordinates_velocity=None,
    additional_parameters_values=None,
    basis_tolerance=1e-6,
    number_basis=None,
    size_batch=10_000,
    number_worker=8,
    hankel=True,
    los_definition="bisector",
    kmin=None,
    **kwargs,
):
    """Reduced power spectrum basis and its flip covariances.

    flip covariances are linear in the input power spectra. Each input
    spectrum ``s`` is decomposed as ``P_s(k; theta) ~ sum_n a_sn(theta) b_sn(k)``
    with an SVD of the training spectra. The covariance of every term is then
    ``C_t(theta) = sum_n a_sn(theta) C_t[b_sn]`` with ``s`` the spectrum of
    term ``t``. The ``C_t[b_sn]`` are computed here, once, with flip.

    Args:
        power_spectrum_list: List of flip power spectrum dictionaries, one per
            parameter sample.
        parameter_values: Array ``(n_samples, n_params)``.
        model_name: flip analytical model name.
        covariance_type: Block type ``("gg"|"vv"|"gv")``.
        coordinates_density: Density coordinates.
        coordinates_velocity: Velocity coordinates.
        additional_parameters_values: Extra parameters of the model kernels
            (fixed: they are not emulated).
        basis_tolerance: Keep the components with singular value above
            ``basis_tolerance`` times the largest one.
        number_basis: Number of components; overrides ``basis_tolerance``.
        size_batch: Batch size of the flip covariance computation.
        number_worker: Number of workers of the flip covariance computation.
        hankel: Use the Hankel transform (flip default).
        los_definition: Line-of-sight convention.
        kmin: Low-k regularisation of flip.
        **kwargs: Unused.

    Returns:
        Tuple ``(square_covariance, coefficients, basis_covariances,
        spectrum_indices, basis_dictionary)``:

        - ``coefficients``: array ``(n_spectra, n_samples, n_basis)``;
        - ``basis_covariances``: array ``(n_terms, n_basis, n_flat)`` of flat
          covariances;
        - ``spectrum_indices``: spectrum index of each term;
        - ``basis_dictionary``: wavenumbers, basis and singular values.
    """
    if len(power_spectrum_list) != len(parameter_values):
        raise ValueError(
            "power_spectrum_list and parameter_values must have the same length."
        )
    square_covariance = is_square_covariance(covariance_type)

    spectra_samples = [
        power_spectrum_dict[covariance_type]
        for power_spectrum_dict in power_spectrum_list
    ]
    number_spectra = len(spectra_samples[0])
    wavenumber = np.asarray(spectra_samples[0][0][0])
    for spectra in spectra_samples:
        for k, _ in spectra:
            if not np.allclose(k, wavenumber):
                raise ValueError(
                    "All the power spectra of a block must share the same k grid."
                )

    # One SVD per input spectrum: spectra of very different amplitudes keep
    # their own relative accuracy.
    bases, singular_values = [], []
    for s in range(number_spectra):
        stacked = np.array([np.asarray(spectra[s][1]) for spectra in spectra_samples])
        _, singular_values_s, basis_s = np.linalg.svd(stacked, full_matrices=False)
        bases.append(basis_s)
        singular_values.append(singular_values_s)

    if number_basis is None:
        number_basis = max(
            int(np.sum(singular_values_s / singular_values_s[0] > basis_tolerance))
            for singular_values_s in singular_values
        )
    number_basis = min(number_basis, min(len(basis_s) for basis_s in bases))
    bases = np.array([basis_s[:number_basis] for basis_s in bases])

    coefficients = np.array(
        [
            [np.asarray(spectra[s][1]) @ bases[s].T for spectra in spectra_samples]
            for s in range(number_spectra)
        ]
    )

    # Every spectrum s gets its n-th basis function: one flip call per n gives
    # C_t[b_{s(t) n}] for all terms t.
    basis_covariances = []
    for n in range(number_basis):
        basis_covariances.append(
            flip_generator.compute_cov(
                model_name,
                covariance_type,
                [[wavenumber, bases[s][n]] for s in range(number_spectra)],
                coordinates_density=coordinates_density,
                coordinates_velocity=coordinates_velocity,
                additional_parameters_values=additional_parameters_values,
                size_batch=size_batch,
                number_worker=number_worker,
                hankel=hankel,
                los_definition=los_definition,
                kmin=kmin,
            )
        )
    basis_covariances = np.swapaxes(np.array(basis_covariances), 0, 1)

    log.add(
        f"Power spectrum basis for {covariance_type}: {number_basis} components "
        f"per spectrum, {number_spectra} spectra",
        level="info",
    )
    basis_dictionary = {
        "wavenumber": wavenumber,
        "basis": bases,
        "singular_values": singular_values,
    }
    return (
        square_covariance,
        coefficients,
        basis_covariances,
        term_spectrum_indices(model_name, covariance_type),
        basis_dictionary,
    )
