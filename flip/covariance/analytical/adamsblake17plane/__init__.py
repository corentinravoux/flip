"""Adams & Blake (2017), plane-parallel (flat-sky) covariance model.

Flat-sky approximation of :mod:`flip.covariance.analytical.adamsblake17`: the
line of sight is taken as a fixed direction rather than per-pair, which is faster
but valid only for small angular separations. Free parameters and coordinate keys
match the wide-angle model (``fs8``, ``bs8``, ``sigv``; ``ra``, ``dec``,
``rcom_zobs``).
"""

_variant = [None]

_free_par = {"fs8": "velocity@all", "bs8": "density@all", "sigv": "velocity@all"}

_coordinate_keys = ["ra", "dec", "rcom_zobs"]
