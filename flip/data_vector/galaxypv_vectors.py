"""Galaxy peculiar-velocity data vectors from distance indicators.

Data vectors that turn galaxy distance-indicator measurements into peculiar
velocities: log-distance ratios ``eta`` (:class:`VelFromLogDist`), the
Tully-Fisher relation (:class:`VelFromTullyFisher`) and the Fundamental Plane
(:class:`VelFromFundamentalPlane`). Each standardizes its observables into a
distance modulus, forms the Hubble-diagram residual against the fiducial
:math:`5\\log_{10}[(1+z)r(z)/h] + 25`, converts it to a velocity with a
redshift-dependent estimator, and propagates the measurement errors (or a full
observation covariance) plus an intrinsic scatter ``sigma_M``.
"""

from flip.utils import create_log

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

log = create_log()


class VelFromLogDist(DataVector):
    """Velocity vector from log-distance ratios ``eta``.

    The log-distance ratio :math:`\\eta = \\log_{10}(d_z/d_H)` is mapped directly
    to a peculiar velocity through :math:`v = 5\\,J(z)\\,\\eta`, where :math:`J(z)`
    is the redshift-dependent estimator selected by ``velocity_estimator``. No
    standardization parameters are fitted.

    Required keys:
        ``eta`` (and ``eta_error`` when no observation covariance is provided).
    """

    _kind = "velocity"
    _needed_keys = ["eta"]
    _free_par = []
    _number_dimension_observation_covariance = 1
    _parameters_observation_covariance = ["eta"]

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
        super().__init__(data, covariance_observation=covariance_observation)

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

        if self._covariance_observation is None:
            velocity_variance = (
                log_distance_to_velocity * self._data["eta_error"]
            ) ** 2

        else:
            conversion_matrix = jnp.diag(log_distance_to_velocity)

            velocity_variance = (
                conversion_matrix @ self._covariance_observation @ conversion_matrix.T
            )

        return velocity, velocity_variance


class VelFromTullyFisher(DataVector):
    """Velocity vector from the Tully-Fisher relation.

    Standardizes the mean apparent magnitude ``m_mean`` against the log line
    width ``logW`` through :math:`\\mu = m_{\\rm mean} + a\\,\\log W + b`, forms the
    Hubble-diagram residual and converts it to a peculiar velocity. Slope ``a``,
    zero-point ``b`` and intrinsic scatter ``sigma_M`` are free parameters.

    Required keys:
        ``zobs``, ``logW``, ``m_mean``, ``rcom_zobs`` (plus ``e_logW``,
        ``e_m_mean`` when no observation covariance is provided).
    """

    _kind = "velocity"
    _needed_keys = ["zobs", "logW", "m_mean", "rcom_zobs"]
    _free_par = ["a", "b"]
    _number_dimension_observation_covariance = 2
    _parameters_observation_covariance = ["logW", "m_mean"]

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

    def __init__(
        self,
        data,
        h,
        covariance_observation=None,
        optional_covariance_observed_distance_modulus=None,
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

        self.optional_covariance_observed_distance_modulus = (
            optional_covariance_observed_distance_modulus
        )
        self.velocity_estimator = velocity_estimator
        if self.optional_covariance_observed_distance_modulus is not None:
            optional_covariance = jnp.array(
                optional_covariance_observed_distance_modulus
            )
            if optional_covariance.shape != (
                self._number_datapoints,
                self._number_datapoints,
            ):
                raise ValueError(
                    f"Optional covariance must be of shape {(self._number_datapoints, self._number_datapoints)}, "
                    f"but got {optional_covariance.shape}."
                )
        self.h = h
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
            if self.optional_covariance_observed_distance_modulus is not None:
                variance_distance_modulus = (
                    jnp.diag(variance_distance_modulus)
                    + self.optional_covariance_observed_distance_modulus
                )
        else:
            weights_observation_covariance = jnp.array(
                [
                    1.0,
                    parameter_values_dict["a"],
                ]
            )
            jacobian = jnp.kron(
                weights_observation_covariance,
                jnp.eye(self._number_datapoints),
            )
            variance_distance_modulus = (
                jacobian @ self._covariance_observation @ jacobian.T
            )
            variance_distance_modulus += (
                jnp.eye(self._number_datapoints) * parameter_values_dict["sigma_M"] ** 2
            )

            if self.optional_covariance_observed_distance_modulus is not None:
                variance_distance_modulus += (
                    self.optional_covariance_observed_distance_modulus
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
            if self.optional_covariance_observed_distance_modulus is not None:

                velocity_variance = (
                    distance_modulus_difference_to_velocity[:, None]
                    * observed_distance_modulus_variance
                    * distance_modulus_difference_to_velocity[None, :]
                )
            else:
                velocity_variance = (
                    observed_distance_modulus_variance
                    * distance_modulus_difference_to_velocity**2
                )
        else:

            velocity_variance = (
                distance_modulus_difference_to_velocity[:, None]
                * observed_distance_modulus_variance
                * distance_modulus_difference_to_velocity[None, :]
            )

        velocities = (
            distance_modulus_difference_to_velocity
            * self.compute_distance_modulus_difference(parameter_values_dict)
        )

        if self._host_matrix is not None:
            velocities, velocity_variance = vector_utils.get_grouped_data_variance(
                self._host_matrix, velocities, velocity_variance
            )

        return velocities, velocity_variance


class VelFromFundamentalPlane(DataVector):
    """Velocity vector from the Fundamental Plane relation.

    Standardizes early-type galaxies through the Fundamental Plane
    :math:`\\mu = 5(\\log R_e - a\\,\\log\\sigma - b\\,\\log I - c)`, forms the
    Hubble-diagram residual and converts it to a peculiar velocity. Plane
    coefficients ``a``, ``b``, ``c`` and intrinsic scatter ``sigma_M`` are free
    parameters.

    Required keys:
        ``zobs``, ``logRe``, ``logsig``, ``logI``, ``rcom_zobs`` (plus
        ``e_logRe``, ``e_logsig``, ``e_logI`` when no observation covariance is
        provided).
    """

    _kind = "velocity"
    _needed_keys = ["zobs", "logRe", "logsig", "logI", "rcom_zobs"]
    _free_par = ["a", "b", "c"]
    _number_dimension_observation_covariance = 3
    _parameters_observation_covariance = ["logRe", "logsig", "logI"]

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

    def __init__(
        self,
        data,
        h,
        covariance_observation=None,
        optional_covariance_observed_distance_modulus=None,
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
        self.optional_covariance_observed_distance_modulus = (
            optional_covariance_observed_distance_modulus
        )
        if self.optional_covariance_observed_distance_modulus is not None:
            optional_covariance = jnp.array(
                optional_covariance_observed_distance_modulus
            )
            if optional_covariance.shape != (
                self._number_datapoints,
                self._number_datapoints,
            ):
                raise ValueError(
                    f"Optional covariance must be of shape {(self._number_datapoints, self._number_datapoints)}, "
                    f"but got {optional_covariance.shape}."
                )
        self.h = h
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

            if self.optional_covariance_observed_distance_modulus is not None:
                variance_distance_modulus = (
                    jnp.diag(variance_distance_modulus)
                    + self.optional_covariance_observed_distance_modulus
                )
        else:
            weights_observation_covariance = jnp.array(
                [
                    1.0,
                    parameter_values_dict["a"],
                    parameter_values_dict["b"],
                ]
            )
            jacobian = jnp.kron(
                weights_observation_covariance,
                jnp.eye(self._number_datapoints),
            )
            variance_distance_modulus = (
                jacobian @ self._covariance_observation @ jacobian.T
            )
            variance_distance_modulus += (
                jnp.eye(self._number_datapoints) * parameter_values_dict["sigma_M"] ** 2
            )
            if self.optional_covariance_observed_distance_modulus is not None:
                variance_distance_modulus += (
                    self.optional_covariance_observed_distance_modulus
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
            if self.optional_covariance_observed_distance_modulus is not None:
                velocity_variance = (
                    distance_modulus_difference_to_velocity[:, None]
                    * observed_distance_modulus_variance
                    * distance_modulus_difference_to_velocity[None, :]
                )
            else:
                velocity_variance = (
                    observed_distance_modulus_variance
                    * distance_modulus_difference_to_velocity**2
                )
        else:
            velocity_variance = (
                distance_modulus_difference_to_velocity[:, None]
                * observed_distance_modulus_variance
                * distance_modulus_difference_to_velocity[None, :]
            )
        velocities = (
            distance_modulus_difference_to_velocity
            * self.compute_distance_modulus_difference(parameter_values_dict)
        )

        if self._host_matrix is not None:
            velocities, velocity_variance = vector_utils.get_grouped_data_variance(
                self._host_matrix, velocities, velocity_variance
            )

        return velocities, velocity_variance
