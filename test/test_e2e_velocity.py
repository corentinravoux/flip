import json
from pathlib import Path

import numpy as np
import pandas as pd

from flip import __flip_dir_path__, data_vector, fitter, utils


def test_e2e_velocity_fit_short():
    base_path = Path(__flip_dir_path__)
    data_path = base_path / "data"

    # Load packaged tiny dataset and subsample further for speed
    velocity_df = pd.read_parquet(data_path / "velocity_data.parquet")
    # Rename and build true-velocity dictionary with zero observational noise
    velocity_dict = velocity_df.rename(columns={"vpec": "velocity"}).to_dict(
        orient="list"
    )
    sample_size = min(60, len(velocity_dict["velocity"]))
    selection = np.arange(sample_size)
    velocity_dict = {k: np.asarray(v)[selection] for k, v in velocity_dict.items()}
    velocity_dict["velocity_error"] = np.zeros(sample_size)

    velocity_data_vector = data_vector.DirectVel(velocity_dict)

    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    sigu = 15.0
    power_spectra = {"vv": [[ktt, ptt * utils.Du(ktt, sigu) ** 2]]}

    covariance = velocity_data_vector.compute_covariance(
        "carreres23", power_spectra, size_batch=2000, number_worker=1
    )

    like_props = {"inversion_method": "cholesky"}
    params = {
        "fs8": {"value": 0.4, "limit_low": 0.0, "limit_up": 1.0, "fixed": False},
        "sigv": {"value": 50.0, "limit_low": 0.0, "limit_up": 500.0, "fixed": False},
    }

    fit_minuit = fitter.FitMinuit.init_from_covariance(
        covariance,
        velocity_data_vector,
        params,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=like_props,
    )

    # Run optimization twice to check reproducibility
    vals1 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)

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
