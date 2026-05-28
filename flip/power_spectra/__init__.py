"""Init file of the flip.power_spectra package."""

from flip._subpackages import require

require("power_spectra")

from . import models
from .generator import compute_power_spectra
