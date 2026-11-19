import types

import numpy as np
import pytest
from flip.covariance import covariance, fitter
from flip.covariance.emulators import generator, prepare
from flip.data import load_data_test

from flip import data_vector, utils

pytest.importorskip("sklearn")


def _velocity_setup(n):
    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(subsample=n)
    _, _, (ktt, ptt) = load_data_test.load_power_spectra()

    def power_spectrum_dict(sigu):
        return {"vv": [[ktt, ptt * utils.Du(ktt, sigu) ** 2]]}

    def exact_covariance(sigu):
        return covariance.CovMatrix.init_from_flip(
            "carreres23",
            "velocity",
            power_spectrum_dict(sigu),
            coordinates_velocity=coordinates_velocity,
            size_batch=100_000,
            number_worker=1,
        )

    return coordinates_velocity, velocity_data, power_spectrum_dict, exact_covariance


def _matrix(covariance_object, key, term):
    covariance_object.compute_matrix_covariance(verbose=False)
    return np.asarray(covariance_object.covariance_dict[key][term])


def _relative_error(emulated, exact):
    return np.max(np.abs(np.asarray(emulated) - exact)) / np.max(np.abs(exact))


def test_matrix_evaluation_functions_keep_their_term():
    # Fake backend: variance models are 10 and 20, non-diagonal models 1 and 2.
    fake_module = types.SimpleNamespace(
        evaluate=lambda model, x, dictionary: (
            np.full((1, 1 if model >= 10 else 3), float(model)),
            None,
        )
    )
    models = ([10, 20], [1, 2], [None, None], [None, None])
    functions = generator.return_evaluation_functions(
        models, fake_module, ["sigu"], "vv"
    )
    for j, function in enumerate(functions):
        matrix = function({"sigu": 15.0})
        assert matrix[0, 0] == 10.0 * (j + 1)
        assert matrix[0, 1] == float(j + 1)


def test_prepare_matrix_does_not_modify_covariances():
    _, _, _, exact_covariance = _velocity_setup(20)
    flat_covariance = exact_covariance(15.0)
    flat_covariance.compute_flat_covariance(verbose=False)
    matrix_covariance = exact_covariance(15.0)
    matrix_covariance.compute_matrix_covariance(verbose=False)

    _, variance, non_diagonal = prepare.prepare_matrix(
        [flat_covariance, matrix_covariance], "vv"
    )
    assert flat_covariance.matrix_form is False
    assert matrix_covariance.matrix_form is True
    np.testing.assert_allclose(variance[0, 0], variance[0, 1])
    np.testing.assert_allclose(non_diagonal[0, 0], non_diagonal[0, 1])


def test_unknown_emulator_raises():
    with pytest.raises(ValueError):
        generator.load_emulator_module("unknown")


def test_skgpmatrix_velocity():
    _, _, _, exact_covariance = _velocity_setup(50)
    sigu_train = np.linspace(10.0, 20.0, 10)
    covariance_emulator = covariance.CovMatrix.init_from_emulator(
        "skgpmatrix",
        "velocity",
        [exact_covariance(s) for s in sigu_train],
        sigu_train[:, np.newaxis],
        ["sigu"],
    )
    for sigu in [11.3, 17.9]:
        error = _relative_error(
            covariance_emulator.covariance_dict["vv"][0]({"sigu": sigu}),
            _matrix(exact_covariance(sigu), "vv", 0),
        )
        assert error < 1e-4


def _density_setup(n):
    coordinates_density, _ = load_data_test.load_density_data(subsample=n)
    power_spectrum_dict = {"gg": load_data_test.load_power_spectrum_dict()["gg"]}

    def exact_covariance(sig_g):
        return covariance.CovMatrix.init_from_flip(
            "adamsblake20",
            "density",
            power_spectrum_dict,
            coordinates_density=coordinates_density,
            additional_parameters_values=(sig_g,),
            size_batch=100_000,
            number_worker=1,
        )

    return exact_covariance


@pytest.mark.parametrize("emulator_model_name", ["skgpmatrix", "gpmatrix"])
def test_matrix_multi_term_density(emulator_model_name):
    if emulator_model_name == "gpmatrix":
        pytest.importorskip("GPy")
    exact_covariance = _density_setup(30)
    sig_g_train = np.linspace(1.0, 5.0, 10)
    covariance_emulator = covariance.CovMatrix.init_from_emulator(
        emulator_model_name,
        "density",
        [exact_covariance(s) for s in sig_g_train],
        sig_g_train[:, np.newaxis],
        ["sig_g"],
    )
    exact = exact_covariance(2.7)
    for j, function in enumerate(covariance_emulator.covariance_dict["gg"]):
        assert _relative_error(function({"sig_g": 2.7}), _matrix(exact, "gg", j)) < 1e-2


def test_skgppowerspectrum_velocity():
    coordinates_velocity, _, power_spectrum_dict, exact_covariance = _velocity_setup(50)
    sigu_train = np.linspace(8.0, 22.0, 30)
    covariance_emulator = covariance.CovMatrix.init_from_emulator(
        "skgppowerspectrum",
        "velocity",
        None,
        sigu_train[:, np.newaxis],
        ["sigu"],
        power_spectrum_list=[power_spectrum_dict(s) for s in sigu_train],
        model_name="carreres23",
        coordinates_velocity=coordinates_velocity,
        number_worker=1,
    )
    for sigu in [11.3, 17.9]:
        error = _relative_error(
            covariance_emulator.covariance_dict["vv"][0]({"sigu": sigu}),
            _matrix(exact_covariance(sigu), "vv", 0),
        )
        assert error < 1e-4


def test_skgppowerspectrum_full():
    n = 20
    coordinates_density, _ = load_data_test.load_density_data(subsample=n)
    coordinates_velocity, _ = load_data_test.load_velocity_data(subsample=n)

    def exact_covariance(sigu):
        return covariance.CovMatrix.init_from_flip(
            "adamsblake20",
            "full",
            load_data_test.load_power_spectrum_dict(sigmau_fiducial=sigu),
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            additional_parameters_values=(3.0,),
            size_batch=100_000,
            number_worker=1,
        )

    sigu_train = np.linspace(8.0, 22.0, 20)
    covariance_emulator = covariance.CovMatrix.init_from_emulator(
        "skgppowerspectrum",
        "full",
        None,
        sigu_train[:, np.newaxis],
        ["sigu"],
        power_spectrum_list=[
            load_data_test.load_power_spectrum_dict(sigmau_fiducial=s)
            for s in sigu_train
        ],
        model_name="adamsblake20",
        coordinates_density=coordinates_density,
        coordinates_velocity=coordinates_velocity,
        additional_parameters_values=(3.0,),
        number_worker=1,
    )
    exact = exact_covariance(13.3)
    for key in ["gg", "vv", "gv"]:
        for j, function in enumerate(covariance_emulator.covariance_dict[key]):
            if key == "gv":
                emulated = function({"sigu": 13.3}, n, n)
            else:
                emulated = function({"sigu": 13.3})
            assert _relative_error(emulated, _matrix(exact, key, j)) < 1e-4


def test_skgpscalar_velocity():
    coordinates_velocity, _, power_spectrum_dict, exact_covariance = _velocity_setup(40)
    sigu_train = np.linspace(10.0, 20.0, 10)
    covariance_emulator = covariance.CovMatrix.init_from_emulator(
        "skgpscalar",
        "velocity",
        None,
        sigu_train[:, np.newaxis],
        ["sigu"],
        power_spectrum_list=[power_spectrum_dict(s) for s in sigu_train],
        model_name="carreres23",
        coordinates_velocity=coordinates_velocity,
        number_training_points=400,
    )
    matrix = np.asarray(covariance_emulator.covariance_dict["vv"][0]({"sigu": 14.2}))
    assert _relative_error(matrix, _matrix(exact_covariance(14.2), "vv", 0)) < 1e-2
    np.testing.assert_allclose(matrix, matrix.T)
    assert np.min(np.linalg.eigvalsh(matrix)) > -1e-6 * np.max(np.abs(matrix))


def test_emulator_rejects_jit():
    coordinates_velocity, velocity_data, power_spectrum_dict, _ = _velocity_setup(20)
    sigu_train = np.linspace(10.0, 20.0, 5)
    covariance_emulator = covariance.CovMatrix.init_from_emulator(
        "skgppowerspectrum",
        "velocity",
        None,
        sigu_train[:, np.newaxis],
        ["sigu"],
        power_spectrum_list=[power_spectrum_dict(s) for s in sigu_train],
        model_name="carreres23",
        coordinates_velocity=coordinates_velocity,
        number_worker=1,
    )
    parameter_dict = {
        "fs8": {"value": 0.4, "limit_low": 0.0},
        "sigv": {"value": 200.0, "limit_low": 0.0},
        "sigu": {"value": 15.0, "limit_low": 10.0, "limit_up": 20.0},
    }
    with pytest.raises(ValueError):
        fitter.FitMinuit.init_from_covariance(
            covariance_emulator,
            data_vector.DirectVel(velocity_data),
            parameter_dict,
            likelihood_properties={"inversion_method": "cholesky", "use_jit": True},
        )
