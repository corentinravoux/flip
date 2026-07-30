"""Adams & Blake (2017) covariance model.

Wide-angle (full 3D coordinate) analytical covariance for density and velocity
fields. Free parameters: ``fs8`` (:math:`f\\sigma_8`, velocity), ``bs8``
(:math:`b\\sigma_8`, density) and the velocity dispersion ``sigv``. Coordinates
required per tracer: ``ra``, ``dec``, ``rcom_zobs``. See
:mod:`flip.covariance.analytical.adamsblake17plane` for the flat-sky variant.
"""

_variant = [None]

_free_par = {"fs8": "velocity@all", "bs8": "density@all", "sigv": "velocity@all"}

_coordinate_keys = ["ra", "dec", "rcom_zobs"]
