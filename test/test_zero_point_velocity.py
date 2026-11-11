import numpy as np

from flip import data_vector
from flip.covariance import covariance, fitter
from flip.data import load_data_test
from flip.data_vector import vector_utils


def test_subtract_zero_point_velocity():
    velocities = np.array([100.0, -50.0, 20.0])
    np.testing.assert_array_equal(
        vector_utils.subtract_zero_point_velocity(velocities), velocities
    )
    np.testing.assert_array_equal(
        vector_utils.subtract_zero_point_velocity(velocities, {"fs8": 0.4}),
        velocities,
    )
    np.testing.assert_array_equal(
        vector_utils.subtract_zero_point_velocity(velocities, {"v0": 30.0}),
        velocities - 30.0,
    )


def test_direct_vel_v0():
    _, velocity_data = load_data_test.load_velocity_data(subsample=20)
    vector = data_vector.DirectVel(velocity_data)
    velocity, variance = vector.give_data_and_variance()
    velocity_dict, _ = vector.give_data_and_variance({"fs8": 0.4})
    velocity_v0, variance_v0 = vector.give_data_and_variance({"fs8": 0.4, "v0": 30.0})

    np.testing.assert_array_equal(velocity, velocity_dict)
    np.testing.assert_allclose(velocity_v0, velocity - 30.0)
    np.testing.assert_array_equal(variance_v0, variance)


def test_estimator_v0():
    _, velocity_data = load_data_test.load_velocity_data(subsample=20)
    velocity_data["dmu"] = np.linspace(-0.1, 0.1, 20)
    velocity_data["dmu_error"] = np.full(20, 0.1)
    vector = data_vector.VelFromHDres(velocity_data, velocity_estimator="watkins")

    velocity, variance = vector.give_data_and_variance({"M_0": 0.01})
    velocity_v0, variance_v0 = vector.give_data_and_variance({"M_0": 0.01, "v0": 30.0})

    np.testing.assert_allclose(velocity_v0, velocity - 30.0)
    np.testing.assert_array_equal(variance_v0, variance)


def test_fit_recovers_v0():
    n = 50
    v0_shift = 300.0
    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(subsample=n)
    power_spectrum_dict = load_data_test.load_power_spectrum_dict()
    covariance_object = covariance.CovMatrix.init_from_flip(
        "adamsblake17plane",
        "velocity",
        power_spectrum_dict,
        coordinates_density=None,
        coordinates_velocity=coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
    )

    def fit(velocity_data):
        fit_minuit = fitter.FitMinuit.init_from_covariance(
            covariance_object,
            data_vector.DirectVel(velocity_data),
            {
                "fs8": {
                    "value": 0.4,
                    "limit_low": 0.0,
                    "limit_up": 1.0,
                    "fixed": False,
                },
                "sigv": {
                    "value": 50.0,
                    "limit_low": 0.0,
                    "limit_up": 500.0,
                    "fixed": False,
                },
                "v0": {"value": 0.0, "fixed": False},
            },
            likelihood_type="multivariate_gaussian",
            likelihood_properties={"inversion_method": "cholesky"},
        )
        return fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)

    values = fit(velocity_data)
    shifted_data = dict(velocity_data)
    shifted_data["velocity"] = np.array(velocity_data["velocity"]) + v0_shift
    values_shifted = fit(shifted_data)

    np.testing.assert_allclose(
        values_shifted["v0"] - values["v0"], v0_shift, rtol=0, atol=1.0
    )
    np.testing.assert_allclose(values_shifted["fs8"], values["fs8"], atol=0.01)
    np.testing.assert_allclose(values_shifted["sigv"], values["sigv"], atol=1.0)
