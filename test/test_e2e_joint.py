import json
from pathlib import Path

import numpy as np
import pandas as pd

from flip import __flip_dir_path__, data_vector, fitter, utils


def test_e2e_joint_short():
    base_path = Path(__flip_dir_path__)
    data_path = base_path / "data"

    # Density subset
    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})
    grid = grid.iloc[:30]
    density_data_vector = data_vector.Dens(grid.to_dict(orient="list"))

    # Velocity subset (true velocities, zero noise)
    velocity_df = pd.read_parquet(data_path / "velocity_data.parquet")
    velocity_sample = velocity_df.rename(columns={"vpec": "velocity"}).iloc[:30]
    velocity_dict = velocity_sample.to_dict(orient="list")
    velocity_dict["velocity_error"] = np.zeros(len(velocity_sample))
    velocity_data_vector = data_vector.DirectVel(velocity_dict)

    density_velocity_data_vector = data_vector.DensVel(
        density_data_vector, velocity_data_vector
    )

    # Power spectra
    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")
    kmt, pmt = np.loadtxt(data_path / "power_spectrum_mt.txt")
    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    sigma_u = 15.0
    power_spectra = {
        "gg": [[kmm, pmm]],
        "gv": [[kmt, pmt]],
        "vv": [[ktt, ptt * utils.Du(ktt, sigma_u) ** 2]],
    }

    covariance = density_velocity_data_vector.compute_covariance(
        "adamsblake17plane", power_spectra, size_batch=2000, number_worker=1
    )

    like_props = {"inversion_method": "cholesky"}
    params = {
        "bs8": {"value": 1.0, "limit_low": 0.0, "limit_up": 2.0, "fixed": False},
        "fs8": {"value": 0.4, "limit_low": 0.0, "limit_up": 1.0, "fixed": False},
        "sigv": {"value": 50.0, "limit_low": 0.0, "limit_up": 500.0, "fixed": False},
    }

    fit_minuit = fitter.FitMinuit.init_from_covariance(
        covariance,
        density_velocity_data_vector,
        params,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=like_props,
    )

    vals1 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fit_minuit.run(migrad=True, hesse=False, minos=False, n_iter=1)
    assert 0.2 <= vals1["bs8"] <= 1.8
    assert 0.1 <= vals1["fs8"] <= 0.9
    assert 0.0 <= vals1["sigv"] <= 300.0
    assert abs(vals1["bs8"] - vals2["bs8"]) < 1e-3
    assert abs(vals1["fs8"] - vals2["fs8"]) < 1e-3
    assert abs(vals1["sigv"] - vals2["sigv"]) < 1e-1

    # Compare against saved reference
    with open(data_path / "test_e2e_refs.json", "r") as f:
        refs = json.load(f)["e2e_joint"]
    # Joint fit can drift slightly due to degeneracies; keep tolerances modest
    np.testing.assert_allclose(vals1["bs8"], refs["bs8"], rtol=7e-2, atol=0)
    np.testing.assert_allclose(vals1["fs8"], refs["fs8"], rtol=6e-1, atol=0)
    np.testing.assert_allclose(vals1["sigv"], refs["sigv"], rtol=2e-1, atol=0)
