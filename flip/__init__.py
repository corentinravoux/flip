"""Init file of the flip package."""

import os

from . import covariance, fisher, fitter, gridding, likelihood, power_spectra, utils

__version__ = "1.0.0"
__flip_dir_path__ = os.path.dirname(__file__)
