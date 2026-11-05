from functools import lru_cache

import numpy as np
import pytest
from flip.covariance import covariance, fitter
from flip.covariance.covariance import ProjectedCovariance
from flip.covariance.likelihood import (
    jax_installed,
    log_likelihood_gaussian_cholesky,
    log_likelihood_gaussian_woodbury,
)
from flip.data import load_data_test
from flip.data_vector.vector_utils import redshift_dependence_velocity

from flip import data_vector

_velocity_fiducial = {"fs8": 0.47, "sigv": 150.0}
_velocity_points = [
    {"fs8": 0.40, "sigv": 200.0},
    {"fs8": 0.30, "sigv": 100.0},
    {"fs8": 0.55, "sigv": 250.0},
]
_velocity_params = {
    "fs8": {"value": 0.4, "limit_low": 0.0, "limit_up": 1.0, "fixed": False},
    "sigv": {"value": 200.0, "limit_low": 0.0, "limit_up": 500.0, "fixed": False},
}
_full_fiducial = {"bs8": 1.1, "fs8": 0.47, "beta_f": 0.45, "sigv": 150.0, "sigg": 1.2}
_full_points = [
    {"bs8": 1.0, "fs8": 0.40, "beta_f": 0.40, "sigv": 200.0, "sigg": 1.0},
    {"bs8": 0.8, "fs8": 0.32, "beta_f": 0.30, "sigv": 160.0, "sigg": 0.8},
]
_dense_properties = {"inversion_method": "cholesky"}
_projected_properties = {
    "projection": {"parameter_values_fiducial": _velocity_fiducial, "k": 100}
}


def compare_projected_to_dense(
    projected_covariance, vector, parameter_values_list, vector_variance
):
    """Projected minus dense (Cholesky) log-likelihood at each point."""
    differences = []
    for parameter_values_dict in parameter_values_list:
        dense = log_likelihood_gaussian_cholesky(
            vector,
            projected_covariance.covariance.compute_covariance_sum(
                parameter_values_dict, vector_variance
            ),
        )
        w, noise = projected_covariance.compute_projected_covariance(
            parameter_values_dict, vector_variance
        )
        projected = log_likelihood_gaussian_woodbury(
            projected_covariance.to_frame(vector), projected_covariance.phi, w, noise
        )
        differences.append(float(projected - dense))
    return np.array(differences)


def _velocity_vector(n):
    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(subsample=n)
    # test velocities are exact (velocity_error = 0): use SN-like errors, sigma_mu = 0.15
    velocity_data["velocity_error"] = (
        np.abs(
            redshift_dependence_velocity(
                {"zobs": np.asarray(velocity_data["zobs"])}, "lowz"
            )
        )
        * 0.15
    )
    return coordinates_velocity, data_vector.DirectVel(velocity_data)


@lru_cache
def _velocity_setup(n):
    coordinates_velocity, velocity_vector = _velocity_vector(n)
    covariance_object = covariance.CovMatrix.init_from_flip(
        "carreres23",
        "velocity",
        load_data_test.load_power_spectrum_dict(),
        coordinates_velocity=coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
    )
    vector, variance = velocity_vector.give_data_and_variance()
    return covariance_object, velocity_vector, np.asarray(vector), np.asarray(variance)


@lru_cache
def _full_setup(n):
    coordinates_density, density_data = load_data_test.load_density_data(subsample=n)
    coordinates_velocity, velocity_vector = _velocity_vector(n)
    covariance_object = covariance.CovMatrix.init_from_flip(
        "lai22",
        "full",
        load_data_test.load_power_spectrum_dict(),
        coordinates_density=coordinates_density,
        coordinates_velocity=coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
        variant="baseline",
    )
    vector, variance = data_vector.DensVel(
        data_vector.Dens(density_data), velocity_vector
    ).give_data_and_variance()
    return covariance_object, np.asarray(vector), np.asarray(variance)


def _velocity_likelihood(likelihood_type, likelihood_properties):
    covariance_object, velocity_vector, _, _ = _velocity_setup(100)
    return fitter.FitMinuit.init_from_covariance(
        covariance_object,
        velocity_vector,
        _velocity_params,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
    ).likelihood


def test_projected_exact_at_full_rank_velocity():
    covariance_object, _, vector, variance = _velocity_setup(100)
    projected = ProjectedCovariance(
        covariance_object, _velocity_fiducial, variance, k=vector.size
    )
    np.testing.assert_allclose(
        compare_projected_to_dense(projected, vector, _velocity_points, variance), 0.0, atol=1e-6
    )


def test_projected_exact_at_full_rank_full():
    covariance_object, vector, variance = _full_setup(60)
    projected = ProjectedCovariance(
        covariance_object, _full_fiducial, variance, k=vector.size
    )
    assert len(projected.terms) == 30
    np.testing.assert_allclose(
        compare_projected_to_dense(projected, vector, _full_points, variance), 0.0, atol=1e-6
    )


def test_projected_accuracy_velocity():
    covariance_object, _, vector, variance = _velocity_setup(500)
    projected = ProjectedCovariance(
        covariance_object, _velocity_fiducial, variance, snr_min=0.01
    )
    assert projected.k < vector.size
    differences = compare_projected_to_dense(projected, vector, _velocity_points, variance)
    # likelihood shape: the constant offset does not matter, its variation does
    assert np.max(np.abs(differences - differences.mean())) < 0.05


def test_projected_remainder_nonnegative_at_fiducial():
    covariance_object, _, variance = _full_setup(60)
    projected = ProjectedCovariance(covariance_object, _full_fiducial, variance, k=40)
    remainder = np.asarray(
        projected.coefficients(_full_fiducial) @ projected.remainder_terms
    )
    assert remainder.min() > -1e-8 * remainder.max()


def test_projected_design_points_basis():
    covariance_object, _, _, variance = _velocity_setup(100)
    single = ProjectedCovariance(
        covariance_object, _velocity_fiducial, variance, snr_min=0.01
    )
    several = ProjectedCovariance(
        covariance_object,
        [_velocity_fiducial, {"fs8": 0.3, "sigv": 50.0}],
        [variance, variance],
        snr_min=0.01,
    )
    assert several.k >= single.k


def _correlated_variance(variance, seed=3):
    # diagonal errors plus a few correlated (systematic-like) modes
    modes = np.random.default_rng(seed).normal(0, 1, (variance.size, 3))
    return np.diag(variance) + 0.3 * np.median(variance) * modes @ modes.T


def test_projected_exact_at_full_rank_2d_variance():
    covariance_object, _, vector, variance = _velocity_setup(100)
    variance_2d = _correlated_variance(variance)
    projected = ProjectedCovariance(
        covariance_object, _velocity_fiducial, variance_2d, k=vector.size
    )
    np.testing.assert_allclose(
        compare_projected_to_dense(projected, vector, _velocity_points, variance_2d),
        0.0,
        atol=1e-6,
    )


def test_projected_2d_variance_through_fitter():
    covariance_object, velocity_vector, _, variance = _velocity_setup(100)
    data = {"velocity": np.asarray(velocity_vector.give_data_and_variance()[0])}
    data_2d = data_vector.DirectVel(
        data, covariance_observation=_correlated_variance(variance)
    )
    likelihoods = [
        fitter.FitMinuit.init_from_covariance(
            covariance_object,
            data_2d,
            _velocity_params,
            likelihood_type=likelihood_type,
            likelihood_properties=properties,
        ).likelihood
        for likelihood_type, properties in (
            ("multivariate_gaussian", _dense_properties),
            ("multivariate_gaussian_projected", _projected_properties),
        )
    ]
    for point in _velocity_points:
        x = np.array([point["fs8"], point["sigv"]])
        np.testing.assert_allclose(likelihoods[1](x), likelihoods[0](x), rtol=1e-10, atol=1e-6)


class _ParameterDependentVariance(data_vector.DirectVel):
    def give_data_and_variance(self, parameter_values_dict, *args):
        vector, variance = super().give_data_and_variance()
        return vector, variance * parameter_values_dict["fs8"]


def test_projected_rejects_parameter_dependent_2d_variance():
    covariance_object, velocity_vector, _, variance = _velocity_setup(100)
    data = {"velocity": np.asarray(velocity_vector.give_data_and_variance()[0])}
    data_2d = _ParameterDependentVariance(
        data, covariance_observation=_correlated_variance(variance)
    )
    with pytest.raises(NotImplementedError):
        fitter.FitMinuit.init_from_covariance(
            covariance_object,
            data_2d,
            _velocity_params,
            likelihood_type="multivariate_gaussian_projected",
            likelihood_properties=_projected_properties,
        )


def test_projected_likelihood_through_fitter():
    dense = _velocity_likelihood("multivariate_gaussian", _dense_properties)
    projected = _velocity_likelihood("multivariate_gaussian_projected", _projected_properties)
    for point in _velocity_points:
        x = np.array([point["fs8"], point["sigv"]])
        np.testing.assert_allclose(projected(x), dense(x), rtol=1e-10, atol=1e-6)


def test_projected_gradient_matches_dense():
    if not jax_installed:
        pytest.skip("JAX not available")
    dense = _velocity_likelihood(
        "multivariate_gaussian", {"use_jit": True, **_dense_properties}
    )
    projected = _velocity_likelihood(
        "multivariate_gaussian_projected", {"use_jit": True, **_projected_properties}
    )
    x = np.array([0.4, 200.0])
    np.testing.assert_allclose(
        np.asarray(projected.likelihood_grad(x)),
        np.asarray(dense.likelihood_grad(x)),
        rtol=1e-6,
    )
