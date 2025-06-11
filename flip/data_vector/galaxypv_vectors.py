import numpy as np

from flip import __use_jax__

from . import vector_utils
from .basic import DataVector

if __use_jax__:
    try:
        import jax.numpy as jnp
        from jax import jit
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
        """
        Returns a list of keys needed for the data vector calculation,
        including any additional keys that are conditionally required.

        Returns:
            list: A list of keys needed for the data vector calculation.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["eta_error"]
        return self._needed_keys + cond_keys

    def give_data_and_variance(self, *args):
        """
        Returns the data and variance for the velocity.

        If the covariance observation is available, it returns the velocity data and the covariance observation.
        Otherwise, it returns the velocity data and the squared velocity error.

        Parameters:
            *args: Additional arguments (not used in this method).

        Returns:
            tuple: A tuple containing the velocity data and the variance.

        """
        if self._covariance_observation is not None:
            J = jnp.diag(self._log_distance_to_velocity)
            velocity_variance = J @ self._covariance_observation @ J.T
            return self._data["velocity"], velocity_variance
        return self._data["velocity"], self._data["velocity_error"] ** 2

    def __init__(
        self, data, covariance_observation=None, velocity_estimator="full", **kwargs
    ):
        """
        Initialize the GalaxypvVectors class.

        Parameters:
        - data: DataFrame containing the data.
        - covariance_observation: Covariance matrix of the observations (optional).
        - velocity_estimator: Velocity estimator method (default: "full").
        - **kwargs: Additional keyword arguments.

        """
        self._log_distance_to_velocity = 5 * (
            vector_utils.redshift_dependence_velocity(
                data, velocity_estimator, **kwargs
            )
        )
        data["velocity"] = self._log_distance_to_velocity * data["eta"]

        if covariance_observation is None:
            data["velocity_error"] = self._log_distance_to_velocity * data["eta_error"]

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
        """
        Returns a list of conditional keys based on the availability of covariance observation.

        If the covariance observation is None, the method adds "e_logW" and "e_m_mean" to the list of conditional keys.

        Returns:
            list: A list of conditional keys.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["e_logW", "e_m_mean"]
        return cond_keys

    def compute_observed_distance_modulus(self, parameter_values_dict):
        """
        Compute the observed distance modulus based on the given parameter values.

        Args:
            parameter_values_dict (dict): A dictionary containing the parameter values.

        Returns:
            float: The observed distance modulus.
        """
        observed_distance_modulus = (
            self._data["m_mean"]
            + parameter_values_dict["a"] * self._data["logW"]
            + parameter_values_dict["b"]
        )

        return observed_distance_modulus

    def compute_distance_modulus_difference(self, parameter_values_dict):
        """
        Compute the difference in distance modulus.

        This method calculates the difference in distance modulus based on the provided parameter values.

        Parameters:
            parameter_values_dict (dict): A dictionary containing the parameter values.

        Returns:
            float: The difference in distance modulus.
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
        """
        Compute the variance of the observed distance modulus.

        Args:
            parameter_values_dict (dict): A dictionary containing parameter values.

        Returns:
            float or ndarray: The variance of the observed distance modulus.
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
        """
        Compute the velocities and velocity variances based on the given parameter values.

        Args:
            parameter_values_dict (dict): A dictionary containing the parameter values.

        Returns:
            tuple: A tuple containing the velocities and velocity variances.
        """
        observed_distance_modulus_variance = (
            self.compute_observed_distance_modulus_variance(parameter_values_dict)
        )
        if self._covariance_observation is None:
            velocity_variance = (
                observed_distance_modulus_variance
                * self._distance_modulus_difference_to_velocity**2
            )
        else:
            A = self._init_A()
            J = A[0] + parameter_values_dict["a"] * A[1]
            J = jnp.diag(self._distance_modulus_difference_to_velocity) @ J
            velocity_variance = J @ observed_distance_modulus_variance @ J.T

        velocities = (
            self._distance_modulus_difference_to_velocity
            * self.compute_distance_modulus_difference(parameter_values_dict)
        )

        if self._host_matrix is not None:
            velocities, velocity_variance = vector_utils.get_grouped_data_variance(
                self._host_matrix, velocities, velocity_variance
            )

        return velocities, velocity_variance

    def _init_distance_modulus_difference_to_velocity(
        self, velocity_estimator, **kwargs
    ):
        """
        Initializes the distance modulus difference to velocity calculation.

        Args:
            velocity_estimator: The velocity estimator to use.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the redshift dependence velocity calculation.
        """
        return vector_utils.redshift_dependence_velocity(
            self._data, velocity_estimator, **kwargs
        )

    def _init_A(self):
        """
        Initializes the matrix A for the galaxypv_vectors class.

        Returns:
            A (ndarray): The initialized matrix A.
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
        **kwargs,
    ):
        """
        Initialize the GalaxypvVectors class.

        Args:
            data (dict): The data dictionary containing information about the galaxies.
            h (float): The Hubble constant.
            covariance_observation (ndarray, optional): The covariance matrix of the observations. Defaults to None.
            velocity_estimator (str, optional): The velocity estimator. Defaults to "full".
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the shape of the covariance_observation is not (2 * len(data), 2 * len(data)).
        """
        super().__init__(data, covariance_observation=covariance_observation)
        self._distance_modulus_difference_to_velocity = (
            self._init_distance_modulus_difference_to_velocity(
                velocity_estimator, h=h, **kwargs
            )
        )
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
        """
        Returns a list of conditional keys based on the availability of covariance observation.

        If the covariance observation is None, the method adds "e_logW" and "e_m_mean" to the list of conditional keys.

        Returns:
            list: A list of conditional keys.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["e_logRe", "e_logsig", "e_logI"]
        return cond_keys

    def compute_observed_distance_modulus(self, parameter_values_dict):
        """
        Compute the observed distance modulus based on the given parameter values.

        Args:
            parameter_values_dict (dict): A dictionary containing the parameter values.

        Returns:
            float: The observed distance modulus.
        """
        observed_distance_modulus = 5 * (
            self._data["logRe"]
            - parameter_values_dict["a"] * self._data["logsig"]
            - parameter_values_dict["b"] * self._data["logI"]
            - parameter_values_dict["c"]
        )

        return observed_distance_modulus

    def compute_distance_modulus_difference(self, parameter_values_dict):
        """
        Compute the difference in distance modulus.

        This method calculates the difference in distance modulus based on the provided parameter values.

        Parameters:
            parameter_values_dict (dict): A dictionary containing the parameter values.

        Returns:
            float: The difference in distance modulus.
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
        """
        Compute the variance of the observed distance modulus.

        Args:
            parameter_values_dict (dict): A dictionary containing parameter values.

        Returns:
            float or ndarray: The variance of the observed distance modulus.
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
        """
        Compute the velocities and velocity variances based on the given parameter values.

        Args:
            parameter_values_dict (dict): A dictionary containing the parameter values.

        Returns:
            tuple: A tuple containing the velocities and velocity variances.
        """
        observed_distance_modulus_variance = (
            self.compute_observed_distance_modulus_variance(parameter_values_dict)
        )
        if self._covariance_observation is None:
            velocity_variance = (
                observed_distance_modulus_variance
                * self._distance_modulus_difference_to_velocity**2
            )
        else:
            A = self._init_A()
            J = (
                A[0]
                + parameter_values_dict["a"] * A[1]
                + parameter_values_dict["b"] * A[2]
            )
            J = jnp.diag(self._distance_modulus_difference_to_velocity) @ J
            velocity_variance = J @ observed_distance_modulus_variance @ J.T

        velocities = (
            self._distance_modulus_difference_to_velocity
            * self.compute_distance_modulus_difference(parameter_values_dict)
        )

        if self._host_matrix is not None:
            velocities, velocity_variance = vector_utils.get_grouped_data_variance(
                self._host_matrix, velocities, velocity_variance
            )

        return velocities, velocity_variance

    def _init_distance_modulus_difference_to_velocity(
        self, velocity_estimator, **kwargs
    ):
        """
        Initializes the distance modulus difference to velocity calculation.

        Args:
            velocity_estimator: The velocity estimator to use.
            **kwargs: Additional keyword arguments.

        Returns:
            The result of the redshift dependence velocity calculation.
        """
        return vector_utils.redshift_dependence_velocity(
            self._data, velocity_estimator, **kwargs
        )

    def _init_A(self):
        """
        Initializes the matrix A for the galaxypv_vectors class.

        Returns:
            A (ndarray): The initialized matrix A.
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
        **kwargs,
    ):
        """
        Initialize the GalaxypvVectors class.

        Args:
            data (dict): The data dictionary containing information about the galaxies.
            h (float): The Hubble constant.
            covariance_observation (ndarray, optional): The covariance matrix of the observations. Defaults to None.
            velocity_estimator (str, optional): The velocity estimator. Defaults to "full".
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the shape of the covariance_observation is not (2 * len(data), 2 * len(data)).
        """
        super().__init__(data, covariance_observation=covariance_observation)
        self._distance_modulus_difference_to_velocity = (
            self._init_distance_modulus_difference_to_velocity(
                velocity_estimator, h=h, **kwargs
            )
        )
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
