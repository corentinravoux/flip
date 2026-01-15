import logging
import time

import astropy.constants as acst
import matplotlib.pyplot as plt
import numpy as np

_C_LIGHT_KMS_ = acst.c.to("km/s").value


def Du(k, sigmau):
    """Damping function Du(k, sigma_u) = sin(k sigma_u) / (k sigma_u).

    Args:
        k (array-like): Wavenumber values.
        sigmau (float): Velocity dispersion parameter.

    Returns:
        numpy.ndarray: Damping values for each `k`.
    """
    return np.sin(k * sigmau) / (k * sigmau)


def radec2cart(rcom, ra, dec):
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


def cart2radec(x, y, z):
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
    return rcom, ra, dec


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
