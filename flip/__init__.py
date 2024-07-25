"""Init file of the flip package."""

import os

from flip.utils import create_log

log = create_log()
from . import covariance, fisher, fitter, gridding, likelihood, power_spectra, utils

try:
    import jax
except:
    log.add("Jax is not available, loading numpy and scipy instead")

__version__ = "1.0.0"
__flip_dir_path__ = os.path.dirname(__file__)
