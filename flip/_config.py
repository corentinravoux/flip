import os

# Modules read this flag once, at import. To disable JAX, set the
# FLIP_USE_JAX env var to 0, false or no before ``import flip``.
__use_jax__ = os.environ.get("FLIP_USE_JAX", "1").lower() not in ("0", "false", "no")

# Minimal mode: velocity covariance only, for a sparse checkout of
# flip/covariance. Set the FLIP_MINIMAL env var to 1, true or yes before
# ``import flip``.
__minimal__ = os.environ.get("FLIP_MINIMAL", "0").lower() in ("1", "true", "yes")
