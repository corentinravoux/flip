_variant = [None, "baseline", "nobeta"]


_free_par = {
    "fs8": ["velocity@all", "density@nobeta"],
    "bs8": "density@all",
    "sigv": "velocity@all",
    "beta_f": "density@baseline",
    "sigg": "density@all",
}

_coordinate_keys = ["ra", "dec", "rcom_zobs"]
