"""Ravoux et al. (2025) no-anchor covariance model.

Field-level velocity covariance of Ravoux et al. (2025, arXiv:2501.16852). Rather
than fixing an absolute-magnitude anchor, it treats ``H0`` as the fitted velocity
amplitude (together with the velocity dispersion ``sigv``), which removes the
dependence on an external distance-ladder calibration. Coordinates required:
``ra``, ``dec``, ``rcom_zobs``.
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
