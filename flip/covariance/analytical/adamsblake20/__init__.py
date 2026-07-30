"""Adams & Blake (2020) redshift-dependent covariance model.

Redshift-dependent redshift-space-distortion covariance for density and velocity
that includes the RSD parameter :math:`\\beta`. Variants:
``None``/``"baseline"`` fit ``fs8``, ``bs8``, ``sigv`` (and ``beta_f`` for the
baseline density term); ``"nobeta"`` drops the RSD parameter and fits ``fs8`` on
the density side directly. Coordinates required per tracer: ``ra``, ``dec``,
``rcom_zobs``.
"""

_variant = [None, "baseline", "nobeta"]

_free_par = {
    "fs8": ["velocity@all", "density@nobeta"],
    "bs8": "density@all",
    "sigv": "velocity@all",
    "beta_f": "density@baseline",
}

_coordinate_keys = ["ra", "dec", "rcom_zobs"]
