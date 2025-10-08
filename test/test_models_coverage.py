import numpy as np
import pandas as pd
import pytest
import importlib
from pathlib import Path

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


@pytest.mark.parametrize(
    "model,kind",
    [
        ("carreres23", "velocity"),
        ("adamsblake17plane", "density"),
        ("adamsblake17plane", "velocity"),
        ("adamsblake17plane", "full"),
        ("adamsblake20", "density"),
        ("adamsblake20", "velocity"),
        ("adamsblake20", "full"),
        ("lai22", "density"),
        # rcrk24 is velocity-only (no gg/gv); skip density/full here
        # ravouxcarreres depends on flip_terms; include density/velocity quickly
        ("ravouxcarreres", "density"),
        ("ravouxcarreres", "velocity"),
    ],
)
def test_covariance_generation_shapes(model, kind):
    base = Path(__flip_dir_path__)
    data_path = base / "data"
    (kmm, pmm), (kmt, pmt), (ktt, ptt) = load_ps(data_path)

    sigu = 15.0
    # Some models (adamsblake20, ravouxcarreres) require an extra smoothing parameter sig_g
    needs_sig_g = model in {"adamsblake20", "ravouxcarreres"}
    additional_params = (sigu,) if needs_sig_g else ()

    # Inspect the model's required number of PS terms per block
    flip_terms = importlib.import_module(f"flip.covariance.{model}.flip_terms")
    dictionary_terms = getattr(flip_terms, "dictionary_terms", {})

    def repeat_spec(spec, n):
        # Return a list of n copies of a [k, Pk] pair
        return [[spec[0], spec[1]] for _ in range(max(n, 0))]

    ps = {}
    if kind in ["density", "full", "density_velocity"]:
        if "gg" not in dictionary_terms:
            pytest.xfail(f"Model {model} does not define gg terms (density not supported)")
        ps["gg"] = repeat_spec([kmm, pmm], len(dictionary_terms["gg"]))
    if kind in ["velocity", "full", "density_velocity"]:
        if "vv" not in dictionary_terms:
            pytest.xfail(f"Model {model} does not define vv terms (velocity not supported)")
        ps["vv"] = repeat_spec([ktt, ptt * utils.Du(ktt, sigu) ** 2], len(dictionary_terms["vv"]))
    if kind == "full":
        if "gv" not in dictionary_terms:
            pytest.xfail(f"Model {model} does not define gv terms (full not supported)")
        ps["gv"] = repeat_spec([kmt, pmt], len(dictionary_terms["gv"]))

    if kind == "density":
        dv = small_density(data_path)
    elif kind == "velocity":
        dv = small_velocity_true(data_path)
    else:
        dv = data_vector.DensVel(small_density(data_path), small_velocity_true(data_path))

    cov = dv.compute_covariance(
        model,
        ps,
        size_batch=2000,
        number_worker=1,
        additional_parameters_values=additional_params,
    )

    # Build a parameter dict with reasonable values
    par = {"bs8": 1.0, "fs8": 0.5, "sigv": 0.0}
    if model in {"adamsblake20", "ravouxcarreres"}:
        # Link beta_f consistently to fs8/bs8 so that default and nobeta branches agree
        par["beta_f"] = par["fs8"] / max(par["bs8"], 1e-6)
    if model == "lai22":
        par["sigg"] = 1.0
        # lai22 coefficients expect beta_f unless variant 'nobeta' is used
        par["beta_f"] = par["fs8"] / max(par["bs8"], 1e-6)
    vec, var = dv.give_data_and_variance(par if "M_0" in dv.free_par else {})
    C = cov.compute_covariance_sum(par, var)

    # Check symmetry & finite diagonal
    np.testing.assert_allclose(C, C.T, atol=1e-10)
    assert np.all(np.isfinite(np.diag(C)))
    assert C.shape[0] == vec.shape[0]
