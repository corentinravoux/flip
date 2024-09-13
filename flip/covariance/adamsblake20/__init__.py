_free_par = {
    "density": {"baseline": ["sigv", "bs8", "beta_f"]},
    "velocity": {"baseline": ["fs8", "sigv"]},
}

_free_par["density"]["nobeta"] = _free_par["density"]["baseline"].remove("beta_f")
_free_par["velocity"]["nobeta"] = _free_par["velocity"]["baseline"]

_free_par["density_velocity"] = {
    variant: [i for k in _free_par.keys() for i in _free_par[k][variant]]
    for variant in _free_par["density"].keys()
}

_free_par["full"] = _free_par["density_velocity"]
