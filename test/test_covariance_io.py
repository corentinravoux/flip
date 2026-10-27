import os

import numpy as np
import pytest
from flip.covariance import covariance
from flip.covariance import io as covariance_io
from flip.covariance.likelihood import MultivariateGaussianLikelihood
from flip.data import load_data_test

from flip import data_vector

FORMATS = ["pickle", "npz", "parquet"]
CASES = [
    ("adamsblake17plane", "velocity"),
    ("adamsblake17plane", "density"),
    ("adamsblake17plane", "full"),
    ("carreres23", "velocity"),
]
PARAMETER_VALUES = {"fs8": 0.4, "bs8": 1.0, "sigv": 150.0}
ATTRIBUTES = [
    "model_name",
    "model_kind",
    "free_par",
    "los_definition",
    "variant",
    "number_densities",
    "number_velocities",
    "emulator_flag",
]


@pytest.fixture(scope="module")
def test_data():
    # Unequal sizes so the gg, gv and vv blocks differ in shape.
    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(subsample=30)
    coordinates_density, density_data = load_data_test.load_density_data(subsample=40)
    return {
        "coordinates_velocity": coordinates_velocity,
        "coordinates_density": coordinates_density,
        "velocity": data_vector.DirectVel(velocity_data),
        "density": data_vector.Dens(density_data),
        "power_spectrum_dict": load_data_test.load_power_spectrum_dict(),
    }


def build(test_data, model, kind):
    return covariance.CovMatrix.init_from_flip(
        model,
        kind,
        test_data["power_spectrum_dict"],
        coordinates_density=(
            test_data["coordinates_density"] if kind != "velocity" else None
        ),
        coordinates_velocity=(
            test_data["coordinates_velocity"] if kind != "density" else None
        ),
        size_batch=50_000,
        number_worker=1,
    )


def vector(test_data, kind):
    if kind == "velocity":
        return test_data["velocity"]
    if kind == "density":
        return test_data["density"]
    return data_vector.DensVel(test_data["density"], test_data["velocity"])


def minus_log_likelihood(cov, data):
    names = list(dict.fromkeys(cov.free_par + data.free_par))
    likelihood = MultivariateGaussianLikelihood(
        covariance=cov,
        data=data,
        parameter_names=names,
        likelihood_properties={"inversion_method": "cholesky"},
    )
    return float(likelihood(np.array([PARAMETER_VALUES[n] for n in names])))


@pytest.mark.parametrize("file_format", FORMATS)
@pytest.mark.parametrize("form", ["matrix", "flat"])
@pytest.mark.parametrize("model, kind", CASES)
def test_write_read_round_trip(tmp_path, test_data, model, kind, form, file_format):
    original = build(test_data, model, kind)
    if form == "flat":
        original.compute_flat_covariance(verbose=False)
    filename = str(tmp_path / f"{model}_{kind}_{form}")

    original.write(filename, file_format)
    loaded = covariance.CovMatrix.init_from_file(filename, file_format)

    assert loaded.matrix_form
    assert loaded.coefficients.__name__ == original.coefficients.__name__
    for attribute in ATTRIBUTES:
        assert getattr(loaded, attribute) == getattr(original, attribute), attribute
    if form == "flat":
        loaded.compute_flat_covariance(verbose=False)
    assert set(loaded.covariance_dict) == set(original.covariance_dict)
    for key in original.covariance_dict:
        np.testing.assert_array_equal(
            np.asarray(loaded.covariance_dict[key]),
            np.asarray(original.covariance_dict[key]),
        )

    data = vector(test_data, kind)
    assert minus_log_likelihood(loaded, data) == minus_log_likelihood(
        build(test_data, model, kind), data
    )


@pytest.mark.parametrize("file_format", FORMATS)
def test_prefactor_dict_round_trip(tmp_path, test_data, file_format):
    original = build(test_data, "carreres23", "velocity")
    original.covariance_prefactor_dict = {
        "redshift_velocity": np.linspace(0.01, 0.1, 30),
        "f_z": np.full(30, 0.52),
    }
    filename = str(tmp_path / "prefactor")
    original.write(filename, file_format)
    loaded = covariance.CovMatrix.init_from_file(filename, file_format)
    assert set(loaded.covariance_prefactor_dict) == set(
        original.covariance_prefactor_dict
    )
    for key, value in original.covariance_prefactor_dict.items():
        np.testing.assert_array_equal(loaded.covariance_prefactor_dict[key], value)


@pytest.mark.parametrize("file_format", FORMATS)
def test_non_numeric_array_is_refused(tmp_path, test_data, file_format):
    original = build(test_data, "adamsblake17plane", "velocity")
    original.covariance_prefactor_dict = {"bad": lambda z: z}
    with pytest.raises(TypeError):
        original.write(str(tmp_path / "refused"), file_format)
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize("file_format", FORMATS)
def test_failed_write_leaves_no_file(tmp_path, test_data, monkeypatch, file_format):
    original = build(test_data, "adamsblake17plane", "velocity")

    def fail_after_partial_write(*args, **kwargs):
        file = next(a for a in list(args) + list(kwargs.values()) if hasattr(a, "write"))
        file.write(b"partial")
        raise RuntimeError("disk full")

    if file_format == "pickle":
        monkeypatch.setattr(covariance_io.pickle, "dump", fail_after_partial_write)
    elif file_format == "npz":
        monkeypatch.setattr(covariance_io.np, "savez", fail_after_partial_write)
    else:
        import pyarrow.parquet

        monkeypatch.setattr(pyarrow.parquet, "write_table", fail_after_partial_write)
    with pytest.raises(RuntimeError):
        original.write(str(tmp_path / "failed"), file_format)
    assert os.listdir(tmp_path) == []


def test_unknown_format(tmp_path, test_data):
    original = build(test_data, "adamsblake17plane", "velocity")
    with pytest.raises(ValueError):
        original.write(str(tmp_path / "x"), "hdf5")
    with pytest.raises(ValueError):
        covariance.CovMatrix.init_from_file(str(tmp_path / "x"), "hdf5")
