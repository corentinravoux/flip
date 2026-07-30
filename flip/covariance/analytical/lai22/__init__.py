"""Lai et al. (2022) covariance model.

Analytical density+velocity covariance following Lai et al. (2022). Variants:
``None``/``"baseline"`` fit ``fs8``, ``bs8``, ``sigv`` (and ``beta_f`` for the
baseline density term); ``"nobeta"`` fits ``fs8`` directly on the density side.
Coordinates required per tracer: ``ra``, ``dec``, ``rcom_zobs``.
"""

_variant = [None, "baseline", "nobeta"]


_free_par = {
    "fs8": ["velocity@all", "density@nobeta"],
    "bs8": "density@all",
    "sigv": "velocity@all",
    "beta_f": "density@baseline",
}

_coordinate_keys = ["ra", "dec", "rcom_zobs"]
