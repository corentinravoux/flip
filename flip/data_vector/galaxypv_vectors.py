import numpy as np

from . import vector_utils
from .basic import DataVector

try:
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    jax_installed = True
except ImportError:
    import numpy as jnp

    jax_installed = False


class VelFromLogDist(DataVector):
    _needed_keys = ["eta"]

    @property
    def conditional_needed_keys(self):
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["eta_error"]
        return self._needed_keys + cond_keys

    def __init__(self, data, covariance_observation=None):

        self._log_distance_to_velocity = vector_utils.redshift_dependence_log_distance(
            data
        )

        data["velocity"] = self._log_distance_to_velocity * data["eta"]

        if covariance_observation is not None:
            J = jnp.diag(self._log_distance_to_velocity)
            covariance_observation = J @ covariance_observation @ J.T
        else:
            data["velocity_error"] = self._log_distance_to_velocity * data["eta_error"]
        super().__init__(
            data,
            covariance_observation=covariance_observation,
        )
