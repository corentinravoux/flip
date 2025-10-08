from pathlib import Path

import numpy as np
import pandas as pd

from flip import __flip_dir_path__, data_vector, utils


def test_density_velocity_block_diagonal_no_gv():
    base = Path(__flip_dir_path__)
    data_path = base / "data"

    # Build small density and velocity data vectors from packaged parquet
    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})
    grid = grid.iloc[:5]
    dens = data_vector.Dens(grid.to_dict(orient="list"))

    vdf = pd.read_parquet(data_path / "velocity_data.parquet")
    v = vdf.rename(columns={"vpec": "velocity"}).iloc[:7]
    v_dict = v.to_dict(orient="list")
    v_dict["velocity_error"] = np.zeros(len(v))
    vel = data_vector.DirectVel(v_dict)

    dv = data_vector.DensVel(dens, vel)

    # Use packaged power spectra (no gv term) and init_from_flip through DataVector
    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")
    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    # DensVel.compute_covariance always builds a 'full' covariance; provide a zero gv term to enforce block-diagonal
    kmt, pmt = np.loadtxt(data_path / "power_spectrum_mt.txt")
    ps = {"gg": [[kmm, pmm]], "vv": [[ktt, ptt]], "gv": [[kmt, 0 * pmt]]}

    cov = dv.compute_covariance(
        "adamsblake17plane", ps, size_batch=1000, number_worker=1
    )

    # Assemble total covariance with simple coefficients
    coeffs = {"bs8": 1.0, "fs8": 1.0, "sigv": 0.0}
    vec, var = dv.give_data_and_variance()
    C = cov.compute_covariance_sum(coeffs, var)

    # Check shapes and that cross-block is exactly zero (no gv provided)
    Ng = len(dens.data["density"])
    Nv = len(vel.data["velocity"])
    assert C.shape == (Ng + Nv, Ng + Nv)
    Cgg = C[:Ng, :Ng]
    Cvv = C[Ng:, Ng:]
    Cgv = C[:Ng, Ng:]
    np.testing.assert_allclose(Cgv, 0.0, atol=0.0)
    assert np.all(np.diag(Cgg) > 0)
    assert np.all(np.diag(Cvv) > 0)


def test_full_with_gv_nonzero():
    base = Path(__flip_dir_path__)
    data_path = base / "data"

    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})
    grid = grid.iloc[:4]
    dens = data_vector.Dens(grid.to_dict(orient="list"))

    vdf = pd.read_parquet(data_path / "velocity_data.parquet")
    v = vdf.rename(columns={"vpec": "velocity"}).iloc[:4]
    v_dict = v.to_dict(orient="list")
    v_dict["velocity_error"] = np.zeros(len(v))
    vel = data_vector.DirectVel(v_dict)

    dv = data_vector.DensVel(dens, vel)

    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")
    kmt, pmt = np.loadtxt(data_path / "power_spectrum_mt.txt")
    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    ps = {"gg": [[kmm, pmm]], "gv": [[kmt, pmt]], "vv": [[ktt, ptt]]}

    cov = dv.compute_covariance(
        "adamsblake17plane", ps, size_batch=1000, number_worker=1
    )

    coeffs = {"bs8": 1.0, "fs8": 1.0, "sigv": 0.0}
    vec, var = dv.give_data_and_variance()
    C = cov.compute_covariance_sum(coeffs, var)

    # Cross block should be non-zero in general
    Ng = len(dens.data["density"])
    Cgv = C[:Ng, Ng:]
    assert np.any(np.abs(Cgv) > 0)
