from .._config import __use_jax__
from . import vector_utils
from .basic import DataVector

if __use_jax__:
    try:
        import jax.numpy as jnp
        from jax.experimental.sparse import BCOO

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:

    import numpy as jnp

    jax_installed = False


class VelFromSALTfit(DataVector):
    _kind = "velocity"
    _needed_keys = ["zobs", "mb", "x1", "c", "rcom_zobs"]
    _free_par = ["alpha", "beta", "M_0", "sigma_M"]

    @property
    def conditional_needed_keys(self):
        """Conditionally required SALT2 error and covariance fields.

        Returns:
            list[str]: Includes errors and covariances when covariance is absent.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["e_mb", "e_x1", "e_c", "cov_mb_x1", "cov_mb_c", "cov_x1_c"]
        return cond_keys

    @property
    def conditional_free_par(self):
        """Conditionally required parameters based on host mass.

        Returns:
            list[str]: Includes `gamma` when `host_logmass` is present.
        """
        _cond_fpar = []
        if "host_logmass" in self.data:
            _cond_fpar += ["gamma"]
        return _cond_fpar

    def compute_observed_distance_modulus(self, parameter_values_dict):
        """Compute observed distance modulus from SALT2 fit parameters.

        Args:
            parameter_values_dict (dict): Includes `alpha`, `beta`, `M_0`, optionally `gamma`.

        Returns:
            ndarray: Distance modulus per object.
        """
        observed_distance_modulus = (
            self._data["mb"]
            + parameter_values_dict["alpha"] * self._data["x1"]
            - parameter_values_dict["beta"] * self._data["c"]
            - parameter_values_dict["M_0"]
        )

        if "host_logmass" in self.data:
            mask = self._data["host_logmass"] > 10
            observed_distance_modulus[mask] += parameter_values_dict["gamma"] / 2
            observed_distance_modulus[~mask] -= parameter_values_dict["gamma"] / 2

        return observed_distance_modulus

    def compute_distance_modulus_difference(self, parameter_values_dict):
        """Compute residual distance modulus relative to cosmological expectation.

        Args:
            parameter_values_dict (dict): SALT2 relation parameters.

        Returns:
            ndarray: Residual distance modulus.
        """
        distance_modulus_difference = self.compute_observed_distance_modulus(
            parameter_values_dict
        )

        if self._host_matrix is not None:
            zobs = self.data["zobs_full"]
            rcom_zobs = self.data["rcom_zobs_full"]
        else:
            zobs = self.data["zobs"]
            rcom_zobs = self.data["rcom_zobs"]

        distance_modulus_difference -= (
            5 * jnp.log10((1 + zobs) * rcom_zobs / self.h) + 25
        )
        return distance_modulus_difference

    def compute_observed_distance_modulus_variance(self, parameter_values_dict):
        """Compute variance/covariance of observed SALT2 distance modulus.

        Args:
            parameter_values_dict (dict): Includes `alpha`, `beta`, and `sigma_M`.

        Returns:
            float|ndarray: Variance or covariance depending on inputs.
        """
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

    def give_data_and_variance(self, parameter_values_dict):
        """Compute velocities and their variance from SALT2 relation.

        Args:
            parameter_values_dict (dict): Relation parameters and `sigma_M`.

        Returns:
            tuple: `(velocities, velocity_variance_or_cov)`.
        """
        observed_distance_modulus_variance = (
            self.compute_observed_distance_modulus_variance(parameter_values_dict)
        )
        distance_modulus_difference_to_velocity = (
            vector_utils.redshift_dependence_velocity(
                self._data, self.velocity_estimator, **parameter_values_dict
            )
        )
        if self._covariance_observation is None:
            velocity_variance = (
                observed_distance_modulus_variance
                * distance_modulus_difference_to_velocity**2
            )
        else:
            A = self._init_A()
            J = (
                A[0]
                + parameter_values_dict["alpha"] * A[1]
                - parameter_values_dict["beta"] * A[2]
            )
            J = jnp.diag(distance_modulus_difference_to_velocity) @ J
            velocity_variance = J @ observed_distance_modulus_variance @ J.T

        velocities = (
            distance_modulus_difference_to_velocity
            * self.compute_distance_modulus_difference(parameter_values_dict)
        )

        if self._host_matrix is not None:
            velocities, velocity_variance = vector_utils.get_grouped_data_variance(
                self._host_matrix, velocities, velocity_variance
            )

        return velocities, velocity_variance

    def _init_A(self):
        """Initialize design matrices for linear covariance propagation.

        Returns:
            ndarray: Matrix A blocks.
        """
        N = len(self._data)
        A = jnp.ones((3, N, 3 * N))
        ij = jnp.ogrid[:N, : 3 * N]
        for k in range(3):
            A[k][ij[1] == 3 * ij[0] + k] = 1
        return A

    def __init__(
        self,
        data,
        h,
        covariance_observation=None,
        velocity_estimator="full",
        mass_step=10,
    ):
        """Initialize SN Ia velocity vector from SALT2 fits.

        Args:
            data (dict): Includes SALT2 parameters and cosmology fields.
            h (float): Little-h scaling for distances.
            covariance_observation (ndarray|None): Optional observation covariance.
            velocity_estimator (str): Estimator name.
            mass_step (float): Threshold for host mass step correction.

        Raises:
            ValueError: If covariance shape is not `3N x 3N` when provided.
        """
        super().__init__(data, covariance_observation=covariance_observation)
        self.velocity_estimator = velocity_estimator
        self.h = h
        self._A = None
        self._host_matrix = None
        self._mass_step = mass_step

        if "host_group_id" in data:
            self._host_matrix, self._data_to_group_mapping = (
                vector_utils.compute_host_matrix(self._data["host_group_id"])
            )
            self._data = vector_utils.format_data_multiple_host(
                self._data, self._host_matrix
            )
            if jax_installed:
                self._host_matrix = BCOO.from_scipy_sparse(self._host_matrix)

        if self._covariance_observation is not None:
            if self._covariance_observation.shape != (3 * len(data), 3 * len(data)):
                raise ValueError("Cov should be 3N x 3N")
            self._A = self._init_A()
