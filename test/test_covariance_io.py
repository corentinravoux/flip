import numpy as np
import pytest
from flip.covariance import covariance
from flip.covariance.likelihood import MultivariateGaussianLikelihood
from flip.data import load_data_test

from flip import data_vector


def build(kind):
    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(
        subsample=30
    )
    coordinates_density, density_data = load_data_test.load_density_data(subsample=40)
    cov = covariance.CovMatrix.init_from_flip(
        "adamsblake17plane",
        kind,
        load_data_test.load_power_spectrum_dict(),
        coordinates_density=None if kind == "velocity" else coordinates_density,
        coordinates_velocity=None if kind == "density" else coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
    )
    vectors = {
        "velocity": data_vector.DirectVel(velocity_data),
        "density": data_vector.Dens(density_data),
    }
    vectors["full"] = data_vector.DensVel(vectors["density"], vectors["velocity"])
    return cov, vectors[kind]


def minus_log_likelihood(cov, vector):
    names = list(dict.fromkeys(cov.free_par + vector.free_par))
    likelihood = MultivariateGaussianLikelihood(
        covariance=cov,
        data=vector,
        parameter_names=names,
        likelihood_properties={"inversion_method": "cholesky"},
    )
    values = {"fs8": 0.4, "bs8": 1.0, "sigv": 150.0}
    return likelihood(np.array([values[name] for name in names]))


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("file_format", ["pickle", "npz", "parquet"])
@pytest.mark.parametrize("kind", ["velocity", "density", "full"])
def test_write_read(tmp_path, kind, file_format, flat):
    reference = minus_log_likelihood(*build(kind))
    cov, vector = build(kind)
    if flat:
        cov.compute_flat_covariance(verbose=False)
    filename = str(tmp_path / kind)

    cov.write(filename, file_format)
    loaded = covariance.CovMatrix.init_from_file(filename, file_format)

    assert minus_log_likelihood(loaded, vector) == reference


def test_unknown_format(tmp_path):
    cov, _ = build("velocity")
    with pytest.raises(ValueError):
        cov.write(str(tmp_path / "x"), "hdf5")
    with pytest.raises(ValueError):
        covariance.CovMatrix.init_from_file(str(tmp_path / "x"), "hdf5")
