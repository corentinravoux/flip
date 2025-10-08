import json
import numpy as np
import pandas as pd

from pathlib import Path

from flip import __flip_dir_path__, utils, data_vector, fitter


def test_e2e_velocity_fit_short():
    base = Path(__flip_dir_path__)
    data_path = base / "data"

    # Load packaged tiny dataset and subsample further for speed
    df = pd.read_parquet(data_path / "velocity_data.parquet")
    # Rename and build true-velocity dict with zero observational noise
    d = df.rename(columns={"vpec": "velocity"}).to_dict(orient="list")
    n = min(60, len(d["velocity"]))
    sel = np.arange(n)
    d = {k: np.asarray(v)[sel] for k, v in d.items()}
    d["velocity_error"] = np.zeros(n)

    dv = data_vector.DirectVel(d)

    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    sigu = 15.0
    ps = {"vv": [[ktt, ptt * utils.Du(ktt, sigu) ** 2]]}

    cov = dv.compute_covariance(
        "carreres23", ps, size_batch=2000, number_worker=1
    )

    like_props = {"inversion_method": "cholesky"}
    params = {
        "fs8": {"value": 0.4, "limit_low": 0.0, "limit_up": 1.0, "fixed": False},
        "sigv": {"value": 50.0, "limit_low": 0.0, "limit_up": 500.0, "fixed": False},
    }

    fm = fitter.FitMinuit.init_from_covariance(
        cov, dv, params, likelihood_type="multivariate_gaussian", likelihood_properties=like_props
    )

    # Run optimization twice to check reproducibility
    vals1 = fm.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fm.run(migrad=True, hesse=False, minos=False, n_iter=1)

    # Constraining assertions
    assert 0.1 <= vals1["fs8"] <= 0.9
    assert 0.0 <= vals1["sigv"] <= 300.0
    # Reproducibility within tight tolerances
    assert abs(vals1["fs8"] - vals2["fs8"]) < 1e-3
    # Allow small absolute wiggle; check relative agreement
    assert abs(vals1["sigv"] - vals2["sigv"]) / max(1.0, abs(vals1["sigv"])) < 1e-3

    # Compare against saved reference values
    with open(data_path / "test_e2e_refs.json", "r") as f:
        refs = json.load(f)["e2e_velocity"]
    np.testing.assert_allclose(vals1["fs8"], refs["fs8"], rtol=5e-3, atol=0)
    np.testing.assert_allclose(vals1["sigv"], refs["sigv"], rtol=5e-2, atol=0)
