"""Init file of the flip package."""
import os
from . import covariance
from . import fitter
from . import gridding
from . import likelihood
from . import model_evaluation
from . import utils
from . import power_spectra


__version__ = "1.0.0"
__flip_dir_path__ =  os.path.dirname(__file__)
