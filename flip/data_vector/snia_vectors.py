from flip.utils import create_log
from .basic import DataVector, redshift_dependence_velocity

try:
    import jax.numpy as jnp
    import jax.scipy as jsc
    from jax import jit

    jax_installed = True
except ImportError:
    import numpy as jnp
    import scipy as jsc

    jax_installed = False


class VelFromSALTfit(DataVector):
    _needed_keys = ["zobs", "mb", "x1", "c", "rcom_zobs"]
    _free_par = ["alpha", "beta", "M_0", "sigma_M"]

    @property
    def conditional_needed_keys(self):
        cond_keys = []
        if self._cov is None:
            cond_keys += ["e_mb", "e_x1", "e_c", "cov_mb_x1", "cov_mb_c", "cov_x1_c"]
        return cond_keys

    def compute_observed_distance_modulus(self, parameter_values_dict):
        mu = self._data["mb"]
        mu += parameter_values_dict["alpha"] * self._data["x1"]
        mu -= parameter_values_dict["beta"] * self._data["c"]
        mu -= parameter_values_dict["M_0"]
        return mu

    def compute_dmu(self,parameter_values_dict):
        dmu = self.compute_observed_distance_modulus(parameter_values_dict)
        dmu -= 5 * jnp.log10((1 + self._data["zobs"]) * self._data["rcom_zobs"]) + 25
        return dmu

    def compute_observed_distance_modulus_error(self, parameter_values_dict):
        if self._cov is None:
            var_mu = (
                self._data["e_mb"] ** 2
                + parameter_values_dict["alpha"] ** 2 * self._data["e_x1"] ** 2
                + parameter_values_dict["beta"] ** 2 * self._data["e_c"] ** 2
            )
            var_mu += (
                2 * parameter_values_dict["alpha"] * self._data["cov_mb_x1"]
                - 2 * parameter_values_dict["beta"] * self._data["cov_mb_c"]
                - 2
                * parameter_values_dict["alpha"]
                * parameter_values_dict["beta"]
                * self._data["cov_x1_c"]
            )
            var_mu += parameter_values_dict["sigma_M"]**2
        else:
            return self._cov + jnp.eye(self._cov.shape[0]) * parameter_values_dict["sigma_M"]**2
        return var_mu

    def _give_data_and_errors(self, parameter_values_dict):
        vel_err = self.compute_observed_distance_modulus_error(parameter_values_dict)
        if self._cov is None:
            vel_err *= self._dmu2vel**2
            vel_err = jnp.sqrt(vel_err)
        else:
            J = A[0] + alpha * A[1] - beta * A[2]
            J *= self._dmu2vel
            vel_err = J @ var_vel @ J.T
        return self._dmu2vel * self.compute_dmu(parameter_values_dict), vel_err

    def _init_dmu2vel(self, vel_estimator, **kwargs):
        return redshift_dependence_velocity(self._data, vel_estimator, **kwargs)

    def _init_A(self):
        N = len(self._data)
        A = jnp.ones((3, N, 3 * N))
        ij = jnp.ogrid[:N, : 3 * N]
        for k in range(3):
            A[k][ij[1] == 3 * ij[0] + k] = 1
        return A

    def __init__(self, data, cov=None, vel_estimator="full", **kwargs):
        super().__init__(data, cov=cov)
        self._dmu2vel = self._init_dmu2vel(vel_estimator, **kwargs)

        self._A = None
        if self._cov is not None:
            if self._cov.shape != (3 * len(data), 3 * len(data)):
                raise ValueError("Cov should be 3N x 3N")
            self._A = self._init_A()
