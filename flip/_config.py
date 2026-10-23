import os

# Modules read this flag once, at import. To disable JAX, set the
# FLIP_USE_JAX env var to 0, false or no before ``import flip``.
__use_jax__ = os.environ.get("FLIP_USE_JAX", "1").lower() not in ("0", "false", "no")
