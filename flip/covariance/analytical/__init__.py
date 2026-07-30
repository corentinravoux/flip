"""Analytical covariance model registry.

Each submodule is one covariance model exposing a common interface:
``_free_par`` (parameter -> ``tracer@variant`` map), ``_variant`` (available
variants) and ``_coordinate_keys`` (coordinates required per tracer), plus
``coefficients.py`` (``get_coefficients``) and ``flip_terms.py`` (k-space
multipole kernels). Models range from wide-angle and plane-parallel Adams-Blake
forms to the redshift-dependent ``adamsblake20`` (recommended default),
``lai22``, ``rcrk24``, the Ravoux line (``ravouxcarreres``, ``ravouxnoanchor25``,
``ravouxqin26``) and the ``genericzdep`` template.
"""

from . import (
    adamsblake17,
    adamsblake17plane,
    adamsblake20,
    carreres23,
    genericzdep,
    lai22,
    ravouxcarreres,
    ravouxnoanchor25,
    ravouxqin26,
    rcrk24,
)
