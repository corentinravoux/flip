import numpy as np
from .basic import DataVector, redshift_dependence_velocity
from . import vector_utils as vec_ut

try:
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    jax_installed = True
except ImportError:
    import numpy as jnp

    jax_installed = False


class VelFromSALTfit(DataVector):
    _kind = "velocity"
    _needed_keys = ["zobs", "mb", "x1", "c", "rcom_zobs"]
    _free_par = ["alpha", "beta", "M_0", "sigma_M"]

    @property
    def conditional_needed_keys(self):
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["e_mb", "e_x1", "e_c", "cov_mb_x1", "cov_mb_c", "cov_x1_c"]
        return cond_keys

    def compute_observed_distance_modulus(self, parameter_values_dict):
        observed_distance_modulus = (
            self._data["mb"]
            + parameter_values_dict["alpha"] * self._data["x1"]
            - parameter_values_dict["beta"] * self._data["c"]
            - parameter_values_dict["M_0"]
        )
        return observed_distance_modulus

    def compute_distance_modulus_difference(self, parameter_values_dict, h):
        distance_modulus_difference = self.compute_observed_distance_modulus(
            parameter_values_dict
        )

        if self._host_matrix is not None:
            zobs = self.data["zobs_full"]
            rcom_zobs = self.data["rcom_zobs_full"]
        else:
            zobs = self.data["zobs"]
            rcom_zobs = self.data["rcom_zobs"]

        distance_modulus_difference -= 5 * jnp.log10((1 + zobs) * rcom_zobs / h) + 25
        return distance_modulus_difference

    def compute_observed_distance_modulus_variance(self, parameter_values_dict):
        if self._covariance_observation is None:
            variance_distance_modulus = (
                self._data["e_mb"] ** 2
                + parameter_values_dict["alpha"] ** 2 * self._data["e_x1"] ** 2
                + parameter_values_dict["beta"] ** 2 * self._data["e_c"] ** 2
            )
            variance_distance_modulus += (
                2 * parameter_values_dict["alpha"] * self._data["cov_mb_x1"]
                - 2 * parameter_values_dict["beta"] * self._data["cov_mb_c"]
                - 2
                * parameter_values_dict["alpha"]
                * parameter_values_dict["beta"]
                * self._data["cov_x1_c"]
            )
            variance_distance_modulus += parameter_values_dict["sigma_M"] ** 2
        else:
            variance_distance_modulus = (
                self._covariance_observation
                + jnp.eye(self._covariance_observation.shape[0])
                * parameter_values_dict["sigma_M"] ** 2
            )
        return variance_distance_modulus

    def _give_data_and_variance(self, parameter_values_dict):
        velocity_variance = self.compute_observed_distance_modulus_variance(
            parameter_values_dict
        )
        if self._covariance_observation is None:
            velocity_variance *= self._dmu2vel**2
        else:
            A = self._init_A()
            J = (
                A[0]
                + parameter_values_dict["alpha"] * A[1]
                - parameter_values_dict["beta"] * A[2]
            )
            J *= self._dmu2vel
            velocity_variance = J @ velocity_variance @ J.T

        velocities = self._dmu2vel * self.compute_distance_modulus_difference(
            parameter_values_dict, self.h
        )

        if self._host_matrix is not None:
            velocities, velocity_variance = vec_ut.get_grouped_data_variance(
                self._host_matrix, velocities, velocity_variance
            )

        return (velocities, velocity_variance)

    def _init_dmu2vel(self, vel_estimator, **kwargs):
        return redshift_dependence_velocity(self._data, vel_estimator, **kwargs)

    def _init_A(self):
        N = len(self._data)
        A = jnp.ones((3, N, 3 * N))
        ij = jnp.ogrid[:N, : 3 * N]
        for k in range(3):
            A[k][ij[1] == 3 * ij[0] + k] = 1
        return A

    def __init__(self, data, h, cov=None, vel_estimator="full", **kwargs):
        super().__init__(data, cov=cov)
        self._dmu2vel = self._init_dmu2vel(vel_estimator, h=h, **kwargs)
        self.h = h
        self._A = None
        self._host_matrix = None

        if "host_group_id" in data:
            self._host_matrix, self._data_to_group_mapping = vec_ut.compute_host_matrix(
                self._data["host_group_id"]
            )
            self._data = vec_ut.format_data_multiple_host(self._data, self._host_matrix)
            if jax_installed:
                self._host_matrix = BCOO.from_scipy_sparse(self._host_matrix)

        if self._covariance_observation is not None:
            if self._covariance_observation.shape != (3 * len(data), 3 * len(data)):
                raise ValueError("Cov should be 3N x 3N")
            self._A = self._init_A()
