"""RCRK (2024) growth-index / growth-rate velocity covariance model.

Redshift-dependent velocity covariance that can be parameterized two ways, chosen
by variant: ``"growth_rate"`` fits ``fs8`` (:math:`f\\sigma_8`) directly, while
``"growth_index"`` fits the matter density ``Om0`` and the growth index
``gamma`` (:math:`f = \\Omega_m(z)^\\gamma`), letting one test departures from
the GR growth history. Both fit the velocity dispersion ``sigv``. Coordinates
required: ``ra``, ``dec``, ``rcom_zobs``, ``zobs``.
"""

_variant = ["growth_index", "growth_rate"]

_free_par = {
    "Om0": "velocity@growth_index",
    "gamma": "velocity@growth_index",
    "fs8": "velocity@growth_rate",
    "sigv": "velocity@all",
}

_coordinate_keys = ["ra", "dec", "rcom_zobs", "zobs"]
