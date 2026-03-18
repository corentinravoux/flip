_variant = [
    None,
    "baseline",
    "nosigmavv",
    "biaslink",
    "nosigmavv_biaslink",
]

_free_par = {
    "fs8": ["velocity@all", "density@all"],
    "b1s8": ["density@all"],
    "b2s8": ["density@all"],
    "bss8": ["density@baseline", "density@nosigmavv"],
    "b3nls8": ["density@baseline", "density@nosigmavv"],
    "b1vs8": ["velocity@all"],
    "b2vs8": ["velocity@all"],
    "bsvs8": ["velocity@baseline", "velocity@nosigmavv"],
    "b3nlvs8": ["velocity@baseline", "velocity@nosigmavv"],
    "sigv1sq": ["density@all"],
    "sigv2sq": ["density@all"],
    "sigv3sq": ["density@all"],
    "sigvv1sq": ["velocity@baseline", "velocity@biaslink"],
    "sigvv2sq": ["velocity@baseline", "velocity@biaslink"],
}


_coordinate_keys = ["ra", "dec", "rcom_zobs", "zobs"]
