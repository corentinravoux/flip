"""Generic redshift-dependent covariance model (template).

Template model for building new redshift-dependent covariances: it carries the
extra ``zobs`` coordinate so kernels can depend explicitly on redshift. Ships
with only the velocity dispersion ``sigv`` as a free parameter; copy this
directory and fill in ``coefficients.py`` / ``flip_terms.py`` to add a model.
Coordinates required: ``ra``, ``dec``, ``rcom_zobs``, ``zobs``.
"""

_variant = [None]

_free_par = {"sigv": "velocity@all"}

_coordinate_keys = ["ra", "dec", "rcom_zobs", "zobs"]
