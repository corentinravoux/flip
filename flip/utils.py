import logging
import time

import astropy.constants as acst
import matplotlib.pyplot as plt
import numpy as np

from flip._config import __use_jax__

if __use_jax__:
    try:
        import jax.numpy as jnp
        import jax.scipy as jsc

        jax_installed = True

    except ImportError:
        import numpy as jnp
        import scipy as jsc

        jax_installed = False
else:
    import numpy as jnp

    jax_installed = False


_C_LIGHT_KMS_ = acst.c.to("km/s").value

_available_priors = ["gaussian", "positive", "uniform"]


def Du(k, sigmau):
    """Damping function Du(k, sigma_u) = sin(k sigma_u) / (k sigma_u).

    Args:
        k (array-like): Wavenumber values.
        sigmau (float): Velocity dispersion parameter.

    Returns:
        numpy.ndarray: Damping values for each `k`.
    """
    return np.sin(k * sigmau) / (k * sigmau)


# Tolerance applied on the bounds used to detect inputs given in wrong units.
_ANGLE_UNIT_TOLERANCE_ = 1e-6
_DISTANCE_UNIT_TOLERANCE_ = 0.02


def _to_finite_numpy(array):
    """Convert an input to a flat float numpy array, or None when not possible."""
    try:
        array = np.asarray(array, dtype=float).ravel()
    except Exception:
        return None
    if array.size == 0:
        return None
    return array


def check_angle_units(ra, dec, name="coordinates"):
    """Warn when angular coordinates look like degrees instead of radians.

    flip expects right ascension and declination in radians. A declination
    outside [-pi/2, pi/2] or a right ascension outside [-2pi, 2pi] cannot be
    in radians, so a warning is logged. Catalogs in degrees restricted to a
    very small patch around the origin (|dec| < 1.57 deg and |ra| < 6.28 deg)
    cannot be distinguished from radians and are not detected.

    Args:
        ra (array-like|None): Right ascension.
        dec (array-like|None): Declination.
        name (str): Label of the checked sample, used in the message.

    Returns:
        bool: True when a warning was emitted.
    """
    ra, dec = _to_finite_numpy(ra), _to_finite_numpy(dec)
    reasons = []
    if dec is not None and np.isfinite(dec).any():
        max_dec = np.nanmax(np.abs(dec[np.isfinite(dec)]))
        if max_dec > np.pi / 2 + _ANGLE_UNIT_TOLERANCE_:
            reasons.append(f"max |dec| = {max_dec:.3f} > pi/2")
    if ra is not None and np.isfinite(ra).any():
        max_ra = np.nanmax(np.abs(ra[np.isfinite(ra)]))
        if max_ra > 2 * np.pi + _ANGLE_UNIT_TOLERANCE_:
            reasons.append(f"max |ra| = {max_ra:.3f} > 2pi")
    if len(reasons) == 0:
        return False
    Logger.add(
        f"The angles of the {name} seem to be in degrees ({', '.join(reasons)}). "
        "flip expects ra and dec in radians, convert them with numpy.deg2rad.",
        level="warning",
    )
    return True


def check_distance_units(rcom, zobs, name="coordinates"):
    """Warn when comoving distances look like Mpc instead of Mpc/h.

    In Mpc/h units, the comoving distance is (c / 100) * int_0^z dz' / E(z'),
    which does not depend on h and is always smaller than c * z / 100 since
    E(z) >= 1. When the median of rcom / (c * z / 100) is larger than one,
    the distances cannot be in Mpc/h and a warning is logged. Distances given
    in Mpc are detected this way up to z ~ 1; above, the test is inconclusive
    and nothing is reported.

    Args:
        rcom (array-like|None): Comoving distances.
        zobs (array-like|None): Redshifts associated to the distances.
        name (str): Label of the checked sample, used in the message.

    Returns:
        bool: True when a warning was emitted.
    """
    rcom, zobs = _to_finite_numpy(rcom), _to_finite_numpy(zobs)
    if rcom is None or zobs is None or rcom.shape != zobs.shape:
        return False
    mask = np.isfinite(rcom) & np.isfinite(zobs) & (zobs > 0) & (rcom > 0)
    if not mask.any():
        return False
    ratio = np.median(rcom[mask] / (_C_LIGHT_KMS_ * zobs[mask] / 100.0))
    if ratio <= 1 + _DISTANCE_UNIT_TOLERANCE_:
        return False
    Logger.add(
        f"The comoving distances of the {name} seem to be in Mpc "
        f"(median of rcom / (c z / 100) = {ratio:.3f} > 1). "
        "flip expects comoving distances in Mpc/h, multiply them by h.",
        level="warning",
    )
    return True


def check_coordinates_units(
    ra=None, dec=None, rcom=None, zobs=None, name="coordinates"
):
    """Run the unit checks available for the given coordinates.

    The angle check needs `ra` and/or `dec`. The distance check needs both
    `rcom` and `zobs`: distances alone do not allow telling Mpc from Mpc/h.
    These checks only log warnings and never raise.

    Args:
        ra (array-like|None): Right ascension.
        dec (array-like|None): Declination.
        rcom (array-like|None): Comoving distances.
        zobs (array-like|None): Redshifts.
        name (str): Label of the checked sample, used in the messages.

    Returns:
        tuple[bool, bool]: `(angle_warning, distance_warning)`.
    """
    try:
        angle_warning = check_angle_units(ra, dec, name=name)
        distance_warning = check_distance_units(rcom, zobs, name=name)
    except Exception:
        return False, False
    return angle_warning, distance_warning


def spherical_to_cartesian(ra, dec, rcom):
    """Convert spherical (r, ra, dec) to Cartesian (x, y, z).

    Args:
        rcom (array-like): Comoving distances.
        ra (array-like): Right ascension in radians.
        dec (array-like): Declination in radians.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: x, y, z coordinates.
    """
    x = rcom * np.cos(ra) * np.cos(dec)
    y = rcom * np.sin(ra) * np.cos(dec)
    z = rcom * np.sin(dec)
    return x, y, z


def spherical_to_cartesian_jacobian(ra, dec, rcom):

    if isinstance(rcom, np.ndarray):
        null_component = np.zeros_like(rcom)
    else:
        null_component = 0

    return np.array(
        [
            [
                np.cos(ra) * np.cos(dec),
                -rcom * np.sin(ra) * np.cos(dec),
                -rcom * np.cos(ra) * np.sin(dec),
            ],
            [
                np.sin(ra) * np.cos(dec),
                rcom * np.cos(ra) * np.cos(dec),
                -rcom * np.sin(ra) * np.sin(dec),
            ],
            [np.sin(dec), null_component, rcom * np.cos(dec)],
        ]
    )


def cartesian_to_spherical(x, y, z):
    """Convert Cartesian (x, y, z) to spherical (r, ra, dec).

    Args:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.
        z (array-like): Z coordinates.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: rcom, ra, dec.
    """
    rcom = np.sqrt(x**2 + y**2 + z**2)
    ra = np.arctan2(y, x)
    mask = rcom != 0
    dec = np.zeros(x.shape)
    dec[mask] = np.arcsin(z[mask] / rcom[mask])
    return ra, dec, rcom


def return_key(dictionary, string, default_value):
    """Return value for `string` in `dictionary`, or a default.

    Args:
        dictionary (dict): Input mapping.
        string (str): Key to retrieve.
        default_value (Any): Fallback if key is absent.

    Returns:
        Any: `dictionary[string]` if present, else `default_value`.
    """
    return dictionary[string] if string in dictionary.keys() else default_value


def __secret_logo__(first_album=False):
    """Show the hidden flip WEBP logo.

    Args:
        first_album (bool): Show the first album variant.
    """
    from PIL import Image

    from flip import __flip_dir_path__

    if first_album:
        img = Image.open(f"{__flip_dir_path__}/data/.htmp/flip_first_album.webp")
    else:
        # Load the WEBP image
        img = Image.open(f"{__flip_dir_path__}/data/.htmp/flip_heavy.webp")

    # Display it with matplotlib
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def create_log(log_level="info"):
    """Create and configure a global logger.

    Args:
        log_level (str): One of `info`, `debug`, `warning`.

    Returns:
        Logger: Configured logger wrapper.
    """
    log = Logger(log_level=log_level)
    log.setup_logging()
    return log


_logging_handler = None


class Logger(object):
    def __init__(self, name="Python_Report", log_level="info"):
        """Initialize Logger wrapper.

        Args:
            name (str): Report filename for file logging.
            log_level (str): Logging level name.
        """
        self.name = name
        self.log_level = log_level

    def setup_logging(self):
        """Configure stream logging with formatted timestamps."""
        levels = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }

        logger = logging.getLogger()
        t0 = time.time()

        class Formatter(logging.Formatter):
            def format(self, record):
                s1 = "[ %09.2f ]: " % (time.time() - t0)
                return s1 + logging.Formatter.format(self, record)

        fmt = Formatter(
            fmt="%(asctime)s %(name)-15s %(levelname)-8s %(message)s",
            datefmt="%m-%d %H:%M ",
        )

        global _logging_handler
        if _logging_handler is None:
            _logging_handler = logging.StreamHandler()
            logger.addHandler(_logging_handler)

        _logging_handler.setFormatter(fmt)
        logger.setLevel(levels[self.log_level])

    def setup_report_logging(self):
        """Configure file logging for reports using basicConfig."""
        levels = {
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }
        logging.basicConfig(
            filename=self.name,
            filemode="w",
            level=levels[self.log_level],
            format="%(asctime)s :: %(levelname)s :: %(message)s",
        )

    @staticmethod
    def add(line, level="info"):
        """Log a message at the given level.

        Args:
            line (str): Message to log.
            level (str): One of `info`, `warning`, `debug`.
        """
        if level == "info":
            logging.info(line)
        if level == "warning":
            logging.warning(line)
        if level == "debug":
            logging.debug(line)

    @staticmethod
    def add_array_statistics(arr, char):
        """Log min, max, mean, and std of an array with a label.

        Args:
            arr (array-like): Input array.
            char (str): Label to include in messages.
        """
        if arr is not None:
            Logger.add(f"Min of {char}: {arr.min()}")
            Logger.add(f"Max of {char}: {arr.max()}")
            Logger.add(f"Mean of {char}: {arr.mean()}")
            Logger.add(f"Standard deviation of {char}: {arr.std()}")

    @staticmethod
    def close():
        """Shut down the logging module cleanly."""
        logging.shutdown()


def return_prior(
    parameter_name,
    prior_properties,
):
    if prior_properties["type"].lower() not in _available_priors:
        raise ValueError(
            f"""The prior type {prior_properties["type"]} is not available"""
            f"""Please choose between {_available_priors}"""
        )
    elif prior_properties["type"].lower() == "gaussian":
        prior = GaussianPrior(
            parameter_name=parameter_name,
            prior_mean=prior_properties["mean"],
            prior_standard_deviation=prior_properties["standard_deviation"],
        )
    elif prior_properties["type"].lower() == "positive":
        prior = PositivePrior(
            parameter_name=parameter_name,
        )
    elif prior_properties["type"].lower() == "uniform":
        prior = UniformPrior(
            parameter_name=parameter_name,
            range=prior_properties["range"],
        )

    return prior


class Prior:
    """Base prior class encapsulating parameter-specific priors.

    Attributes:
        parameter_name (str): Name of the parameter this prior applies to.
    """

    def __init__(
        self,
        parameter_name=None,
    ):
        self.parameter_name = parameter_name


class GaussianPrior(Prior):
    r"""Univariate Gaussian prior on a parameter.

    Models $p(\theta) \propto \exp\{-\tfrac{1}{2}[(\theta-\mu)^2/\sigma^2]\}$.
    """

    def __init__(
        self,
        parameter_name=None,
        prior_mean=None,
        prior_standard_deviation=None,
    ):
        super().__init__(parameter_name=parameter_name)
        self.prior_mean = prior_mean
        self.prior_standard_deviation = prior_standard_deviation

    def __call__(
        self,
        parameter_values_dict,
    ):
        """Return Gaussian log-prior for the parameter value.

        Args:
            parameter_values_dict (dict): Map of parameter names to values.

        Returns:
            float: Log-prior value.
        """
        return -0.5 * (
            np.log(2 * jnp.pi * self.prior_standard_deviation**2)
            + (parameter_values_dict[self.parameter_name] - self.prior_mean) ** 2
            / self.prior_standard_deviation**2
        )


class PositivePrior(Prior):
    """Log-prior enforcing parameter positivity via Heaviside function.

    Returns `log(Heaviside(value))`, which is `0` for positive values and `-inf` otherwise.
    """

    def __init__(
        self,
        parameter_name=None,
    ):
        super().__init__(parameter_name=parameter_name)

    def __call__(
        self,
        parameter_values_dict,
    ):
        """Return log-prior that is zero for positive values, -inf otherwise.

        Args:
            parameter_values_dict (dict): Map of parameter names to values.

        Returns:
            float: Log-prior (0 or -inf).
        """
        return jnp.log(jnp.heaviside(parameter_values_dict[self.parameter_name], 0))


class UniformPrior(Prior):
    """Uniform prior over a finite interval for a parameter.

    Uses `scipy.stats.uniform.logpdf` for the specified range.
    """

    def __init__(self, parameter_name=None, range=None):
        super().__init__(parameter_name=parameter_name)
        self.range = range

    def __call__(
        self,
        parameter_values_dict,
    ):
        """Return uniform log-prior over `[range[0], range[1]]`.

        Args:
            parameter_values_dict (dict): Map of parameter names to values.

        Returns:
            float: Log-prior value (constant inside range, -inf outside).
        """
        value = parameter_values_dict[self.parameter_name]
        return jsc.stats.uniform.logpdf(
            value, loc=self.range[0], scale=self.range[1] - self.range[0]
        )


def prior_sum(priors, x):
    """Sum multiple prior contributions.

    Args:
        priors (list[Callable]): List of prior callables accepting a parameter dict.
        x (dict): Parameter values dictionary.

    Returns:
        float: Sum of all prior log-probabilities.
    """
    return sum(prior(x) for prior in priors)
