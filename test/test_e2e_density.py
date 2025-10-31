import json
from pathlib import Path

import numpy as np
import pandas as pd

from flip import __flip_dir_path__, data_vector, fitter


def test_e2e_density_short():
    base_path = Path(__flip_dir_path__)
    data_path = base_path / "data"

    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})

    # Keep a tiny subset for speed
    n = min(50, len(grid))
    grid = grid.iloc[:n]

    density_data_vector = data_vector.Dens(grid.to_dict(orient="list"))

    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")

    ps = {"gg": [[kmm, pmm]]}

    covariance = density_data_vector.compute_covariance(
        "adamsblake17plane", ps, size_batch=2000, number_worker=1
    )

    # Basic checks: covariance sum is finite and symmetric, with positive diagonal
    vals = {"bs8": 1.0, "fs8": 0.5, "sigv": 0.0}
    data_values, data_variance = density_data_vector.give_data_and_variance()
    covariance_matrix = covariance.compute_covariance_sum(vals, data_variance)
    assert covariance_matrix.shape == (n, n)
    np.testing.assert_allclose(covariance_matrix, covariance_matrix.T, atol=1e-12)
    assert np.all(np.diag(covariance_matrix) > 0)

    # Evaluate simple likelihood (no fit) to ensure plumbing works
    like_props = {"inversion_method": "cholesky"}
    params = {"bs8": {"value": 1.0, "limit_low": 0.0, "fixed": False}}
    fit_minuit = fitter.FitMinuit.init_from_covariance(
        covariance,
        density_data_vector,
        params,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=like_props,
    )
    vals1 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    assert 0.2 <= vals1["bs8"] <= 1.8
    assert abs(vals1["bs8"] - vals2["bs8"]) < 1e-3

    # Compare against saved reference
    with open(data_path / "test_e2e_refs.json", "r") as f:
        refs = json.load(f)["e2e_density"]
    np.testing.assert_allclose(vals1["bs8"], refs["bs8"], rtol=5e-3, atol=0)
