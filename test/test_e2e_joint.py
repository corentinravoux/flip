import json
from pathlib import Path

import numpy as np
import pandas as pd

from flip import __flip_dir_path__, data_vector, fitter, utils


def test_e2e_joint_short():
    base = Path(__flip_dir_path__)
    data_path = base / "data"

    # Density subset
    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})
    grid = grid.iloc[:30]
    dens = data_vector.Dens(grid.to_dict(orient="list"))

    # Velocity subset (true velocities, zero noise)
    vdf = pd.read_parquet(data_path / "velocity_data.parquet")
    v = vdf.rename(columns={"vpec": "velocity"}).iloc[:30]
    v_dict = v.to_dict(orient="list")
    v_dict["velocity_error"] = np.zeros(len(v))
    vel = data_vector.DirectVel(v_dict)

    dv = data_vector.DensVel(dens, vel)

    # Power spectra
    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")
    kmt, pmt = np.loadtxt(data_path / "power_spectrum_mt.txt")
    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    sigu = 15.0
    ps = {
        "gg": [[kmm, pmm]],
        "gv": [[kmt, pmt]],
        "vv": [[ktt, ptt * utils.Du(ktt, sigu) ** 2]],
    }

    cov = dv.compute_covariance(
        "adamsblake17plane", ps, size_batch=2000, number_worker=1
    )

    like_props = {"inversion_method": "cholesky"}
    params = {
        "bs8": {"value": 1.0, "limit_low": 0.0, "limit_up": 2.0, "fixed": False},
        "fs8": {"value": 0.4, "limit_low": 0.0, "limit_up": 1.0, "fixed": False},
        "sigv": {"value": 50.0, "limit_low": 0.0, "limit_up": 500.0, "fixed": False},
    }

    fm = fitter.FitMinuit.init_from_covariance(
        cov,
        dv,
        params,
        likelihood_type="multivariate_gaussian",
        likelihood_properties=like_props,
    )

    vals1 = fm.run(migrad=True, hesse=False, minos=False, n_iter=1)
    vals2 = fm.run(migrad=True, hesse=False, minos=False, n_iter=1)
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
