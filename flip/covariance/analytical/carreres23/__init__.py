"""Carreres et al. (2023) velocity covariance model.

Velocity-only analytical covariance (arXiv:2303.01198), the peculiar-velocity
model flip was originally built on. Free parameters: ``fs8``
(:math:`f\\sigma_8`) and the velocity dispersion ``sigv``. Coordinates required:
``ra``, ``dec``, ``rcom_zobs``. Exposes :func:`covariance_vv` from its generator.
"""

from .generator import covariance_vv

_variant = [None]


_free_par = {"fs8": "velocity@all", "sigv": "velocity@all"}

_coordinate_keys = ["ra", "dec", "rcom_zobs"]
