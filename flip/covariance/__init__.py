"""Init file of the flip.covariance package.

In minimal mode (``FLIP_MINIMAL``, see :mod:`flip._config`) only the
modules building a velocity covariance are imported: no fitter,
likelihood, Fisher, emulator or symbolic code.
"""

from flip._config import __minimal__
from flip._subpackages import require

require("covariance")

if __minimal__:
    from . import analytical, cov_utils, generator
else:
    from . import (
        analytical,
        contraction,
        cov_utils,
        emulators,
        fisher,
        fitter,
        generator,
        likelihood,
        symbolic,
    )
from .covariance import CovMatrix
