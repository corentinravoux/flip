"""Init file of the flip package.

Subpackages are loaded lazily on attribute access. Doing
``import flip`` probes dependencies and prints an availability banner
but does not import any subpackage. Doing ``import flip.forward``
loads only ``flip.forward`` and its dependencies, leaving the rest of
the package untouched.

Set the ``FLIP_QUIET`` env var to any value to suppress the banner.
"""

import importlib
import os

from flip.utils import create_log

from . import _subpackages
from .utils import __secret_logo__

log = create_log()

try:
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:
    log.add("Jax is not available, loading numpy and scipy instead")

__flip_dir_path__ = os.path.dirname(__file__)
__version__ = "1.2.3"


_LAZY_SUBPACKAGES = set(_subpackages.SUBPACKAGES)


def __getattr__(name):
    if name in _LAZY_SUBPACKAGES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + list(_LAZY_SUBPACKAGES))


if not os.environ.get("FLIP_QUIET"):
    print(_subpackages.format_banner(__version__))
