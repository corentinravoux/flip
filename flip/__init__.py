"""Init file of the flip package."""

import os

from flip.utils import create_log

log = create_log()
from . import covariance, data, data_vector, power_spectra, utils
from .utils import __secret_logo__

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except:
    log.add("Jax is not available, loading numpy and scipy instead")

__flip_dir_path__ = os.path.dirname(__file__)

__version__ = "1.2.1"
