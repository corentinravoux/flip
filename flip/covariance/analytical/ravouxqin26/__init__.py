"""Ravoux & Qin (2026) covariance model.

Most recent flip covariance model, integrating non-linear damping directly in the
covariance and a full second-order bias expansion for both density and velocity
tracers (linear ``b1``, quadratic ``b2``, tidal ``bs``, third-order non-local
``b3nl``, all times :math:`\\sigma_8`), plus per-order velocity-dispersion terms.
Variants select which bias/dispersion terms are free: ``"baseline"`` enables the
full expansion, ``"nosigmavv"`` drops the velocity-velocity dispersion,
``"biaslink"`` links density and velocity biases, and combinations thereof.
Coordinates required per tracer: ``ra``, ``dec``, ``rcom_zobs``, ``zobs``.
"""

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
