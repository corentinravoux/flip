import numpy as np
from flip.covariance import covariance, fitter
from flip.data import load_data_test

from flip import data_vector


def test_e2e_density(debug_return=False):
    n = 50
    coordinates_density, density_data = load_data_test.load_density_data(subsample=n)

    density_data_vector = data_vector.Dens(density_data)

    power_spectrum_dict = load_data_test.load_power_spectrum_dict()

    covariance_object = covariance.CovMatrix.init_from_flip(
        "adamsblake17plane",
        "density",
        power_spectrum_dict,
        coordinates_density=coordinates_density,
        coordinates_velocity=None,
        size_batch=50_000,
        number_worker=1,
    )

    vals = {"bs8": 1.0, "fs8": 0.5, "sigv": 0.0}
    _, data_variance = density_data_vector.give_data_and_variance()
    covariance_matrix = covariance_object.compute_covariance_sum(vals, data_variance)

    like_props = {"inversion_method": "cholesky"}
    params = {"bs8": {"value": 1.0, "limit_low": 0.0, "fixed": False}}
    fit_minuit = fitter.FitMinuit.init_from_covariance(
        covariance_object,
        density_data_vector,
        params,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=like_props,
    )
    vals1 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    if debug_return:
        return vals1

    reference_values = load_data_test.load_e2e_test_reference_values()["e2e_density"]

    assert covariance_matrix.shape == (n, n)
    np.testing.assert_allclose(covariance_matrix, covariance_matrix.T, atol=1e-12)
    assert np.all(np.diag(covariance_matrix) > 0)
    assert abs(vals1["bs8"] - vals2["bs8"]) < 0.05
    np.testing.assert_allclose(vals1["bs8"], reference_values["bs8"], rtol=5e-3, atol=0)
