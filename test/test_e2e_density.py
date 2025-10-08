import json
import numpy as np
import pandas as pd
from pathlib import Path

from flip import __flip_dir_path__, data_vector, fitter


def test_e2e_density_short():
    base = Path(__flip_dir_path__)
    data_path = base / "data"

    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})

    # Keep a tiny subset for speed
    n = min(50, len(grid))
    grid = grid.iloc[:n]

    dens = data_vector.Dens(grid.to_dict(orient="list"))

    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")

    ps = {"gg": [[kmm, pmm]]}

    cov = dens.compute_covariance(
        "adamsblake17plane", ps, size_batch=2000, number_worker=1
    )

    # Basic checks: covariance sum is finite and symmetric, with positive diagonal
    vals = {"bs8": 1.0, "fs8": 0.5, "sigv": 0.0}
    vec, var = dens.give_data_and_variance()
    C = cov.compute_covariance_sum(vals, var)
    assert C.shape == (n, n)
    np.testing.assert_allclose(C, C.T, atol=1e-12)
    assert np.all(np.diag(C) > 0)

    # Evaluate simple likelihood (no fit) to ensure plumbing works
    like_props = {"inversion_method": "cholesky"}
    params = {"bs8": {"value": 1.0, "limit_low": 0.0, "fixed": False}}
    fm = fitter.FitMinuit.init_from_covariance(
        cov, dens, params, likelihood_type="multivariate_gaussian", likelihood_properties=like_props
    )
    vals1 = fm.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fm.run(migrad=True, hesse=False, minos=False, n_iter=1)
    assert 0.2 <= vals1["bs8"] <= 1.8
    assert abs(vals1["bs8"] - vals2["bs8"]) < 1e-3

    # Compare against saved reference
    with open(data_path / "test_e2e_refs.json", "r") as f:
        refs = json.load(f)["e2e_density"]
    np.testing.assert_allclose(vals1["bs8"], refs["bs8"], rtol=5e-3, atol=0)
