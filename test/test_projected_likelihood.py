from functools import lru_cache

import numpy as np
import pytest
from flip.covariance import covariance, fitter
from flip.covariance.covariance import ProjectedCovariance
from flip.covariance.likelihood import (
    MultivariateGaussianLikelihoodInterpolate1D,
    MultivariateGaussianLikelihoodProjected,
    MultivariateGaussianLikelihoodProjectedInterpolate1D,
    jax_installed,
    log_likelihood_gaussian_cholesky,
    log_likelihood_projected,
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
        projected = log_likelihood_projected(
            vector, parameter_values_dict, vector_variance, projected_covariance
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


def _velocity_likelihood(
    likelihood_type, likelihood_properties, covariance_object=None, data=None
):
    # dense with the CovMatrix, projected with a ProjectedCovariance as covariance
    dense_covariance, velocity_vector, _, _ = _velocity_setup(100)
    return fitter.FitMinuit.init_from_covariance(
        dense_covariance if covariance_object is None else covariance_object,
        velocity_vector if data is None else data,
        _velocity_params,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
    ).likelihood


def _velocity_projection(data=None):
    covariance_object, velocity_vector, _, _ = _velocity_setup(100)
    return ProjectedCovariance.init_from_data(
        covariance_object,
        velocity_vector if data is None else data,
        _velocity_fiducial,
        k=100,
    )


def test_projected_exact_at_full_rank_velocity():
    covariance_object, _, vector, variance = _velocity_setup(100)
    projected = ProjectedCovariance(
        covariance_object, _velocity_fiducial, variance, k=vector.size
    )
    np.testing.assert_allclose(
        compare_projected_to_dense(projected, vector, _velocity_points, variance),
        0.0,
        atol=1e-6,
    )


def test_projected_exact_at_full_rank_full():
    covariance_object, vector, variance = _full_setup(60)
    projected = ProjectedCovariance(
        covariance_object, _full_fiducial, variance, k=vector.size
    )
    assert len(projected.terms) == 30
    np.testing.assert_allclose(
        compare_projected_to_dense(projected, vector, _full_points, variance),
        0.0,
        atol=1e-6,
    )


def test_projected_exact_density_full_rank():
    covariance_object, vector, variance = _full_setup(60)
    projected = ProjectedCovariance(
        covariance_object,
        _full_fiducial,
        variance,
        k=covariance_object.number_velocities,
        exact_density=True,
    )
    assert projected.k == vector.size
    np.testing.assert_allclose(
        compare_projected_to_dense(projected, vector, _full_points, variance),
        0.0,
        atol=1e-6,
    )


def test_projected_exact_density_accuracy():
    covariance_object, vector, variance = _full_setup(200)
    projected = ProjectedCovariance(
        covariance_object, _full_fiducial, variance, snr_min=0.01, exact_density=True
    )
    assert projected.k < vector.size
    differences = compare_projected_to_dense(projected, vector, _full_points, variance)
    assert np.max(np.abs(differences - differences.mean())) < 0.05


def _per_object_setup(model_name, variant):
    # per-object (N x N) coefficients: genericzdep (fixed), rcrk24 (parameter dependent)
    coordinates_velocity, velocity_vector = _velocity_vector(80)
    z = np.asarray(load_data_test.load_velocity_data(subsample=80)[1]["zobs"])
    power_spectrum_dict = load_data_test.load_power_spectrum_dict()
    prefactor = {"redshift_velocity": z}
    if model_name == "genericzdep":
        power_spectrum_dict = {"vv": power_spectrum_dict["vv"] * 3}
        prefactor.update(f_z=0.5 + z, E_z=1 + z, D_growth_z=1 - 0.5 * z)
    covariance_object = covariance.CovMatrix.init_from_flip(
        model_name,
        "velocity",
        power_spectrum_dict,
        coordinates_velocity=coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
        variant=variant,
        covariance_prefactor_dict=prefactor,
        additional_parameters_values=(0.2,) if model_name == "genericzdep" else None,
    )
    vector, variance = velocity_vector.give_data_and_variance()
    return covariance_object, np.asarray(vector), np.asarray(variance)


def test_projected_fixed_per_object_exact_at_full_rank():
    covariance_object, vector, variance = _per_object_setup("genericzdep", None)
    projected = ProjectedCovariance(
        covariance_object, {"sigv": 150.0}, variance, k=vector.size
    )
    points = [{"sigv": 200.0}, {"sigv": 100.0}]
    np.testing.assert_allclose(
        compare_projected_to_dense(projected, vector, points, variance), 0.0, atol=1e-6
    )


@pytest.mark.parametrize(
    "variant, fiducial",
    [
        ("growth_index", {"Om0": 0.3, "gamma": 0.55, "sigv": 150.0}),
        ("growth_rate", {"fs8": 0.0, "sigv": 150.0}),
    ],
)
def test_projected_rejects_parameter_dependent_per_object(variant, fiducial):
    # rcrk24: per-object coefficients that depend on the parameters (fs8 = 0 included)
    covariance_object, _, variance = _per_object_setup("rcrk24", variant)
    with pytest.raises(NotImplementedError):
        ProjectedCovariance(covariance_object, fiducial, variance)


def test_projected_accuracy_velocity():
    covariance_object, _, vector, variance = _velocity_setup(500)
    projected = ProjectedCovariance(
        covariance_object, _velocity_fiducial, variance, snr_min=0.01
    )
    assert projected.k < vector.size
    differences = compare_projected_to_dense(
        projected, vector, _velocity_points, variance
    )
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
        _velocity_likelihood("multivariate_gaussian", _dense_properties, data=data_2d),
        _velocity_likelihood(
            "multivariate_gaussian_projected",
            {},
            _velocity_projection(data_2d),
            data_2d,
        ),
    ]
    for point in _velocity_points:
        x = np.array([point["fs8"], point["sigv"]])
        np.testing.assert_allclose(
            likelihoods[1](x), likelihoods[0](x), rtol=1e-10, atol=1e-6
        )


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
        ProjectedCovariance.init_from_data(
            covariance_object, data_2d, _velocity_fiducial
        )


def test_projected_likelihood_through_fitter():
    dense = _velocity_likelihood("multivariate_gaussian", _dense_properties)
    projected = _velocity_likelihood(
        "multivariate_gaussian_projected", {}, _velocity_projection()
    )
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
        "multivariate_gaussian_projected", {"use_jit": True}, _velocity_projection()
    )
    x = np.array([0.4, 200.0])
    np.testing.assert_allclose(
        np.asarray(projected.likelihood_grad(x)),
        np.asarray(dense.likelihood_grad(x)),
        rtol=1e-6,
    )


@pytest.mark.parametrize("exact", [True, False])
def test_projected_interpolation_matches_interpolate1d(exact):
    # sig_g enters inside the density terms: one covariance per grid point
    coordinates_density, density_data = load_data_test.load_density_data(subsample=40)
    coordinates_velocity, velocity_vector = _velocity_vector(40)
    data = data_vector.DensVel(data_vector.Dens(density_data), velocity_vector)
    grid = np.array([0.5, 1.0, 2.0, 4.0])
    covariances = [
        covariance.CovMatrix.init_from_flip(
            "adamsblake20",
            "full",
            load_data_test.load_power_spectrum_dict(),
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            size_batch=50_000,
            number_worker=1,
            additional_parameters_values=(sig_g,),
        )
        for sig_g in grid
    ]
    names = ["bs8", "fs8", "beta_f", "sigv", "sig_g"]
    fiducial = {"bs8": 1.1, "fs8": 0.47, "beta_f": 0.45, "sigv": 150.0, "sig_g": 1.2}
    # basis built at the grid point nearest the fiducial sig_g, shared by the others
    options = {"k": 80} if exact else {}  # k = 80: every mode
    reference = ProjectedCovariance.init_from_data(
        covariances[1], data, fiducial, **options
    )
    projections = [
        (
            reference
            if i == 1
            else ProjectedCovariance.init_from_data(c, data, fiducial, basis=reference)
        )
        for i, c in enumerate(covariances)
    ]
    likelihoods = [
        likelihood_class(
            covariance=covariance_list,
            data=data,
            parameter_names=names,
            likelihood_properties=properties,
            interpolation_value_name="sig_g",
            interpolation_value_range=grid,
        )
        for likelihood_class, covariance_list, properties in (
            (
                MultivariateGaussianLikelihoodInterpolate1D,
                covariances,
                _dense_properties,
            ),
            (MultivariateGaussianLikelihoodProjectedInterpolate1D, projections, {}),
        )
    ]
    points = [
        np.array([1.0, 0.4, 0.4, 200.0, 0.7]),
        np.array([0.8, 0.3, 0.3, 120.0, 1.6]),
        np.array([1.2, 0.5, 0.5, 250.0, 3.0]),
    ]
    dense = np.array([likelihoods[0](x) for x in points])
    assert np.ptp(dense) > 1  # the points, sig_g included, change the likelihood
    differences = np.array([likelihoods[1](x) for x in points]) - dense
    if exact:
        np.testing.assert_allclose(differences, 0.0, atol=1e-6)
    else:
        assert np.max(np.abs(differences - differences.mean())) < 0.05


def test_projected_likelihood_rejects_dense_covariance():
    covariance_object, velocity_vector, _, _ = _velocity_setup(100)
    with pytest.raises(TypeError):
        MultivariateGaussianLikelihoodProjected(
            covariance=covariance_object,
            data=velocity_vector,
            parameter_names=["fs8", "sigv"],
        )


def test_projected_interpolation_rejects_unshared_basis():
    _, velocity_vector, _, _ = _velocity_setup(100)
    with pytest.raises(ValueError):
        MultivariateGaussianLikelihoodProjectedInterpolate1D(
            covariance=[_velocity_projection(), _velocity_projection()],
            data=velocity_vector,
            parameter_names=["fs8", "sigv", "sig_g"],
            interpolation_value_name="sig_g",
            interpolation_value_range=np.array([0.5, 1.0]),
        )


def test_select_likelihood_rejects_unknown_type():
    with pytest.raises(ValueError):
        fitter.BaseFitter.select_likelihood("multivariate_gaussian_interp2d")
