import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import importlib

from flip import __flip_dir_path__, utils, data_vector


def load_ps(data_path: Path):
    kmm, pmm = np.loadtxt(data_path / "power_spectrum_mm.txt")
    kmt, pmt = np.loadtxt(data_path / "power_spectrum_mt.txt")
    ktt, ptt = np.loadtxt(data_path / "power_spectrum_tt.txt")
    return (kmm, pmm), (kmt, pmt), (ktt, ptt)


def small_density(data_path: Path, n=16):
    grid = pd.read_parquet(data_path / "density_data.parquet")
    grid = grid.rename(columns={"density_err": "density_error", "rcom": "rcom_zobs"})
    grid = grid.iloc[:n]
    return data_vector.Dens(grid.to_dict(orient="list"))


def small_velocity_true(data_path: Path, n=16):
    vdf = pd.read_parquet(data_path / "velocity_data.parquet")
    v = vdf.rename(columns={"vpec": "velocity"}).iloc[:n]
    d = v.to_dict(orient="list")
    d["velocity_error"] = np.zeros(len(v))
    return data_vector.DirectVel(d)


def build_ps(model: str, kind: str, data_path: Path):
    (kmm, pmm), (kmt, pmt), (ktt, ptt) = load_ps(data_path)
    sigu = 15.0
    flip_terms = importlib.import_module(f"flip.covariance.{model}.flip_terms")
    dictionary_terms = getattr(flip_terms, "dictionary_terms", {})

    def repeat(spec, n):
        return [[spec[0], spec[1]] for _ in range(max(n, 0))]

    ps = {}
    if kind in ["density", "full", "density_velocity"] and "gg" in dictionary_terms:
        ps["gg"] = repeat([kmm, pmm], len(dictionary_terms.get("gg", [])))
    if kind in ["velocity", "full", "density_velocity"] and "vv" in dictionary_terms:
        ps["vv"] = repeat([ktt, ptt * utils.Du(ktt, sigu) ** 2], len(dictionary_terms.get("vv", [])))
    if kind == "full" and "gv" in dictionary_terms:
        ps["gv"] = repeat([kmt, pmt], len(dictionary_terms.get("gv", [])))
    return ps


def compute_cov_metrics(model: str, kind: str):
    base = Path(__flip_dir_path__)
    data_path = base / "data"

    if kind == "density":
        dv = small_density(data_path)
    elif kind == "velocity":
        dv = small_velocity_true(data_path)
    else:
        dv = data_vector.DensVel(small_density(data_path), small_velocity_true(data_path))

    additional_params = ()
    par = {"bs8": 1.0, "fs8": 0.5, "sigv": 0.0}
    if model in {"adamsblake20", "ravouxcarreres"}:
        par["beta_f"] = par["fs8"] / max(par["bs8"], 1e-6)
        additional_params = (15.0,)  # sig_g
    if model == "lai22":
        par["sigg"] = 1.0
        par["beta_f"] = par["fs8"] / max(par["bs8"], 1e-6)

    ps = build_ps(model, kind, data_path)
    cov = dv.compute_covariance(
        model, ps, size_batch=2000, number_worker=1, additional_parameters_values=additional_params
    )
    vec, var = dv.give_data_and_variance(par if "M_0" in dv.free_par else {})
    C = cov.compute_covariance_sum(par, var)
    return {
        "shape": [int(C.shape[0]), int(C.shape[1])],
        "trace": float(np.trace(C)),
        "diag_mean": float(np.mean(np.diag(C))),
        "entry_0_0": float(C[0, 0]),
        "entry_0_-1": float(C[0, -1]),
        "entry_mid_mid": float(C[C.shape[0] // 2, C.shape[1] // 2]),
    }


@pytest.mark.parametrize(
    "model,kind",
    [
        ("carreres23", "velocity"),
        ("adamsblake17plane", "density"),
        ("adamsblake17plane", "full"),
        ("adamsblake20", "density"),
        ("ravouxcarreres", "velocity"),
        ("ravouxcarreres", "density"),
        ("lai22", "density"),
    ],
)
def test_covariance_reference_metrics(model, kind):
    # Load stored reference metrics
    ref_path = Path(__flip_dir_path__) / "data" / "test_cov_refs.json"
    with open(ref_path, "r") as f:
        refs = json.load(f)

    key = f"{model}:{kind}"
    assert key in refs, f"Missing reference metrics for {key}"
    ref = refs[key]

    got = compute_cov_metrics(model, kind)

    # Compare metrics with reasonable tolerances
    assert got["shape"] == ref["shape"]
    np.testing.assert_allclose(got["trace"], ref["trace"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["diag_mean"], ref["diag_mean"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["entry_0_0"], ref["entry_0_0"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["entry_0_-1"], ref["entry_0_-1"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got["entry_mid_mid"], ref["entry_mid_mid"], rtol=1e-6, atol=1e-8)
