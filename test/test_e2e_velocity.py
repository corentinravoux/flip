import numpy as np
from flip.covariance import covariance, fitter
from flip.data import load_data_test

from flip import data_vector


def test_e2e_velocity(debug_return=False):
    n = 50
    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(subsample=n)
    velocity_data_vector = data_vector.DirectVel(velocity_data)

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

    like_props = {"inversion_method": "cholesky"}
    params = {
        "fs8": {"value": 0.4, "limit_low": 0.0, "limit_up": 1.0, "fixed": False},
        "sigv": {"value": 50.0, "limit_low": 0.0, "limit_up": 500.0, "fixed": False},
    }

    fit_minuit = fitter.FitMinuit.init_from_covariance(
        covariance_object,
        velocity_data_vector,
        params,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=like_props,
    )

    # Run optimization twice to check reproducibility
    vals1 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)

    if debug_return:
        return vals1

    assert abs(vals1["fs8"] - vals2["fs8"]) < 0.05
    assert abs(vals1["sigv"] - vals2["sigv"]) < 1.0

    reference_values = load_data_test.load_e2e_test_reference_values()["e2e_velocity"]
    np.testing.assert_allclose(vals1["fs8"], reference_values["fs8"], rtol=0.05, atol=0)
    np.testing.assert_allclose(
        vals1["sigv"], reference_values["sigv"], rtol=1.0, atol=0
    )
