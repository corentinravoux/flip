import os

import numpy as np
from flip import fitter, utils
from flip.covariance import covariance
from pkg_resources import resource_filename

flip_base = resource_filename("flip", ".")
data_files = os.path.join(flip_base, "data")

print(flip_base, data_files)
