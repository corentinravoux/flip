_variant = ["growth_index", "growth_rate"]

_free_par = {
    "Om0": "velocity@growth_index",
    "gamma": "velocity@growth_index",
    "fs8": "velocity@growth_rate",
    "sigv": "velocity@all",
}

_coordinate_keys = ["ra", "dec", "rcom_zobs", "zobs"]
