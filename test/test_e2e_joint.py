import numpy as np
from flip.covariance import covariance, fitter
from flip.data import load_data_test

from flip import data_vector


def test_e2e_joint(debug_return=False):
    n = 50
    coordinates_density, density_data = load_data_test.load_density_data(subsample=n)
    density_data_vector = data_vector.Dens(density_data)

    coordinates_velocity, velocity_data = load_data_test.load_velocity_data(subsample=n)
    velocity_data_vector = data_vector.DirectVel(velocity_data)

    density_velocity_data_vector = data_vector.DensVel(
        density_data_vector, velocity_data_vector
    )
    power_spectrum_dict = load_data_test.load_power_spectrum_dict()

    covariance_object = covariance.CovMatrix.init_from_flip(
        "adamsblake17plane",
        "full",
        power_spectrum_dict,
        coordinates_density=coordinates_density,
        coordinates_velocity=coordinates_velocity,
        size_batch=50_000,
        number_worker=1,
    )

    like_props = {"inversion_method": "cholesky"}
    params = {
        "bs8": {"value": 1.0, "limit_low": 0.0, "limit_up": 2.0, "fixed": False},
        "fs8": {"value": 0.4, "limit_low": 0.0, "limit_up": 1.0, "fixed": False},
        "sigv": {"value": 50.0, "limit_low": 0.0, "limit_up": 500.0, "fixed": False},
    }

    fit_minuit = fitter.FitMinuit.init_from_covariance(
        covariance_object,
        density_velocity_data_vector,
        params,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=like_props,
    )

    vals1 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)

    if debug_return:
        return vals1

    assert abs(vals1["bs8"] - vals2["bs8"]) < 0.05
    assert abs(vals1["fs8"] - vals2["fs8"]) < 0.05
    assert abs(vals1["sigv"] - vals2["sigv"]) < 1.0

    # Compare against saved reference
    reference_values = load_data_test.load_e2e_test_reference_values()["e2e_joint"]
    # Joint fit can drift slightly due to degeneracies; keep tolerances modest

    print(vals1, reference_values)
    np.testing.assert_allclose(vals1["bs8"], reference_values["bs8"], rtol=0.05, atol=0)
    np.testing.assert_allclose(vals1["fs8"], reference_values["fs8"], rtol=0.05, atol=0)
    np.testing.assert_allclose(
        vals1["sigv"], reference_values["sigv"], rtol=1.0, atol=0
    )
