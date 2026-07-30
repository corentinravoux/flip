"""Ravoux & Carreres covariance model.

Combined density+velocity analytical covariance used in the flip development line,
merging the Adams-Blake-style RSD treatment with the Carreres velocity model.
Variants: ``None``/``"baseline"`` fit ``fs8``, ``bs8``, ``sigv`` (and ``beta_f``
for the baseline density term); ``"nobeta"`` fits ``fs8`` on the density side
directly. Coordinates required per tracer: ``ra``, ``dec``, ``rcom_zobs``.
"""

_variant = [None, "baseline", "nobeta"]

_free_par = {
    "fs8": ["velocity@all", "density@nobeta"],
    "bs8": "density@all",
    "sigv": "velocity@all",
    "beta_f": "density@baseline",
}

_coordinate_keys = ["ra", "dec", "rcom_zobs"]
