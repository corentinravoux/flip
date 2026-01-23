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


class VelFromLogDist(DataVector):
    _kind = "velocity"
    _needed_keys = ["eta"]

    @property
    def conditional_needed_keys(self):
        """Conditionally required keys for log-distance estimator.

        Returns:
            list[str]: Includes `eta_error` when covariance is absent.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["eta_error"]
        return self._needed_keys + cond_keys

    def give_data_and_variance(self, parameter_values_dict, *args):
        """Return velocity and variance for log-distance based estimator.

        Args:
            parameter_values_dict (dict): Estimator parameters.

        Returns:
            tuple: `(velocity, covariance_or_variance)`.
        """
        log_distance_to_velocity = 5 * (
            vector_utils.redshift_dependence_velocity(
                self._data, self.velocity_estimator, **parameter_values_dict
            )
        )

        velocity = log_distance_to_velocity * self._data["eta"]

        if self._covariance_observation is not None:
            J = jnp.diag(log_distance_to_velocity)
            velocity_variance = J @ self._covariance_observation @ J.T
            return velocity, velocity_variance

        return velocity, (log_distance_to_velocity * self._data["eta_error"]) ** 2

    def __init__(
        self,
        data,
        covariance_observation=None,
        velocity_estimator="full",
    ):
        """Initialize velocity from log-distance `eta`.

        Args:
            data (dict): Must include `eta` and optionally `eta_error`.
            covariance_observation (ndarray|None): Observed covariance.
            velocity_estimator (str): Estimator name, default `"full"`.
        """
        self.velocity_estimator = velocity_estimator
        super().__init__(
            data,
            covariance_observation=covariance_observation,
        )


class VelFromTullyFisher(DataVector):

    _kind = "velocity"
    _needed_keys = ["zobs", "logW", "m_mean", "rcom_zobs"]
    _free_par = ["a", "b"]

    @property
    def conditional_needed_keys(self):
        """Conditionally required keys when covariance is absent.

        Returns:
            list[str]: Includes `e_logW` and `e_m_mean` when needed.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["e_logW", "e_m_mean"]
        return cond_keys

    def compute_observed_distance_modulus(self, parameter_values_dict):
        """Compute observed distance modulus from Tully–Fisher relation.

        Args:
            parameter_values_dict (dict): Includes `a` and `b`.

        Returns:
            ndarray: Distance modulus per object.
        """
        observed_distance_modulus = (
            self._data["m_mean"]
            + parameter_values_dict["a"] * self._data["logW"]
            + parameter_values_dict["b"]
        )

        return observed_distance_modulus

    def compute_distance_modulus_difference(self, parameter_values_dict):
        """Compute residual distance modulus relative to cosmological expectation.

        Args:
            parameter_values_dict (dict): Includes relation parameters.

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

    def compute_observed_distance_modulus_variance(
        self,
        parameter_values_dict,
    ):
        """Compute variance of the observed distance modulus.

        Args:
            parameter_values_dict (dict): Includes `a` and `sigma_M`.

        Returns:
            float|ndarray: Variance or covariance depending on input.
        """
        if self._covariance_observation is None:
            variance_distance_modulus = (
                self._data["e_m_mean"] ** 2
                + parameter_values_dict["a"] ** 2 * self._data["e_logW"] ** 2
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
        """Compute velocities and their variance from Tully–Fisher relation.

        Args:
            parameter_values_dict (dict): Includes relation parameters and `sigma_M`.

        Returns:
            tuple: `(velocities, velocity_variance_or_cov)`.
        """

        distance_modulus_difference_to_velocity = (
            vector_utils.redshift_dependence_velocity(
                self._data, self.velocity_estimator, **parameter_values_dict
            )
        )
        observed_distance_modulus_variance = (
            self.compute_observed_distance_modulus_variance(parameter_values_dict)
        )
        if self._covariance_observation is None:
            velocity_variance = (
                observed_distance_modulus_variance
                * distance_modulus_difference_to_velocity**2
            )
        else:
            A = self._init_A()
            J = A[0] + parameter_values_dict["a"] * A[1]
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
        """Initialize design matrices for linear propagation with covariance.

        Returns:
            ndarray: Matrix A blocks.
        """
        N = len(self._data)
        A = jnp.ones((2, N, 2 * N))
        ij = jnp.ogrid[:N, : 2 * N]
        for k in range(2):
            A[k][ij[1] == 2 * ij[0] + k] = 1
        return A

    def __init__(
        self,
        data,
        h,
        covariance_observation=None,
        velocity_estimator="full",
    ):
        """Initialize Tully–Fisher velocity vector.

        Args:
            data (dict): Includes `logW`, `m_mean`, redshifts and distances.
            h (float): Little-h scaling for distances.
            covariance_observation (ndarray|None): Optional observation covariance.
            velocity_estimator (str): Estimator name.

        Raises:
            ValueError: If covariance shape is not `2N x 2N` when provided.
        """
        super().__init__(data, covariance_observation=covariance_observation)
        self.velocity_estimator = velocity_estimator
        self.h = h
        self._A = None
        self._host_matrix = None

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
            if self._covariance_observation.shape != (2 * len(data), 2 * len(data)):
                raise ValueError("Cov should be 2N x 2N")


class VelFromFundamentalPlane(DataVector):

    _kind = "velocity"
    _needed_keys = ["zobs", "logRe", "logsig", "logI", "rcom_zobs"]
    _free_par = ["a", "b", "c"]

    @property
    def conditional_needed_keys(self):
        """Conditionally required keys when covariance is absent.

        Returns:
            list[str]: Includes `e_logRe`, `e_logsig`, `e_logI` when needed.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["e_logRe", "e_logsig", "e_logI"]
        return cond_keys

    def compute_observed_distance_modulus(self, parameter_values_dict):
        """Compute observed distance modulus from Fundamental Plane relation.

        Args:
            parameter_values_dict (dict): Includes `a`, `b`, `c`.

        Returns:
            ndarray: Distance modulus per object.
        """
        observed_distance_modulus = 5 * (
            self._data["logRe"]
            - parameter_values_dict["a"] * self._data["logsig"]
            - parameter_values_dict["b"] * self._data["logI"]
            - parameter_values_dict["c"]
        )

        return observed_distance_modulus

    def compute_distance_modulus_difference(self, parameter_values_dict):
        """Compute residual distance modulus relative to cosmological expectation.

        Args:
            parameter_values_dict (dict): Relation parameters.

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

    def compute_observed_distance_modulus_variance(
        self,
        parameter_values_dict,
    ):
        """Compute variance of the observed distance modulus.

        Args:
            parameter_values_dict (dict): Includes `a`, `b`, and `sigma_M`.

        Returns:
            float|ndarray: Variance or covariance depending on input.
        """
        if self._covariance_observation is None:
            variance_distance_modulus = (
                self._data["e_logRe"] ** 2
                + parameter_values_dict["a"] ** 2 * self._data["logsig"] ** 2
                + parameter_values_dict["b"] ** 2 * self._data["logI"] ** 2
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
        """Compute velocities and their variance from Fundamental Plane relation.

        Args:
            parameter_values_dict (dict): Includes relation parameters and `sigma_M`.

        Returns:
            tuple: `(velocities, velocity_variance_or_cov)`.
        """

        distance_modulus_difference_to_velocity = (
            vector_utils.redshift_dependence_velocity(
                self._data, self.velocity_estimator, **parameter_values_dict
            )
        )
        observed_distance_modulus_variance = (
            self.compute_observed_distance_modulus_variance(parameter_values_dict)
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
                + parameter_values_dict["a"] * A[1]
                + parameter_values_dict["b"] * A[2]
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
        """Initialize design matrices for linear propagation with covariance.

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
    ):
        """Initialize Fundamental Plane velocity vector.

        Args:
            data (dict): Includes `logRe`, `logsig`, `logI`, redshifts and distances.
            h (float): Little-h scaling for distances.
            covariance_observation (ndarray|None): Optional observation covariance.
            velocity_estimator (str): Estimator name.

        Raises:
            ValueError: If covariance shape is not `3N x 3N` when provided.
        """
        super().__init__(data, covariance_observation=covariance_observation)
        self.velocity_estimator = velocity_estimator
        self.h = h
        self._A = None
        self._host_matrix = None

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
