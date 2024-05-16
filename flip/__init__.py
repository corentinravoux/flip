"""Init file of the flip package."""
import os

from . import covariance, fitter, gridding, likelihood, utils, power_spectra

__version__ = "1.0.0"
__flip_dir_path__ = os.path.dirname(__file__)
