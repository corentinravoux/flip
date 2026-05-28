"""Init file of the flip.covariance package."""

from flip._subpackages import require

require("covariance")

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
