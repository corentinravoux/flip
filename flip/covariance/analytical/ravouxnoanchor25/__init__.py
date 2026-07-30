"""Ravoux no-anchor velocity covariance model.

Velocity-only variant in the Ravoux line that treats ``H0`` as the fitted
velocity amplitude (together with the velocity dispersion ``sigv``) instead of
fixing an absolute-magnitude anchor, which removes the dependence on an external
distance-ladder calibration. Coordinates required: ``ra``, ``dec``,
``rcom_zobs``.
"""

_variant = [None]


_free_par = {
    "H0": "velocity@all",
    "sigv": "velocity@all",
}

_coordinate_keys = [
    "ra",
    "dec",
    "rcom_zobs",
]
