from flip.utils import create_log

from .._config import __use_jax__
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

log = create_log()


def _variance_from_errors(
    e_mb,
    e_x1,
    e_c,
    cov_mb_x1,
    cov_mb_c,
    cov_x1_c,
    alpha,
    beta,
    sigma_M,
):
    variance = e_mb**2 + alpha**2 * e_x1**2 + beta**2 * e_c**2
    variance += (
        2 * alpha * cov_mb_x1 - 2 * beta * cov_mb_c - 2 * alpha * beta * cov_x1_c
    )
    return variance + sigma_M**2


def _variance_from_covariance(
    covariance_observation,
    number_datapoints,
    alpha,
    beta,
    sigma_M,
):

    weights_observation_covariance = jnp.array(
        [
            1.0,
            alpha,
            -beta,
        ]
    )
    jacobian = jnp.kron(
        weights_observation_covariance,
        jnp.eye(number_datapoints),
    )
    variance_distance_modulus = (
        jacobian @ covariance_observation @ jacobian.T
        + jnp.eye(number_datapoints) * sigma_M**2
    )

    return variance_distance_modulus


if jax_installed:
    _variance_from_errors = jit(_variance_from_errors)
    _variance_from_covariance = jit(_variance_from_covariance)


class VelTrippRelation(DataVector):
    _kind = "velocity"
    _needed_keys = ["zobs", "mb", "x1", "c", "rcom_zobs"]
    _free_par = ["alpha", "beta", "M_0", "sigma_M"]
    _number_dimension_observation_covariance = 3
    _parameters_observation_covariance = ["mb", "x1", "c"]

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

    def __init__(
        self,
        data,
        h,
        covariance_observation=None,
        optional_covariance_observed_distance_modulus=None,
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
            ValueError: If covariance shape is not adapted
        """
        super().__init__(data, covariance_observation=covariance_observation)
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
        self.velocity_estimator = velocity_estimator
        self.h = h
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

    def compute_observed_distance_modulus(self, parameter_values_dict):
        """Compute observed distance modulus from SALT2 fit parameters.

        Args:
            parameter_values_dict (dict): Includes `alpha`, `beta`, `M_0`, optionally `gamma`.

        Returns:
            ndarray: Distance modulus per object.
        """

        if (
            "alpha_low" in parameter_values_dict
            and "alpha_high" in parameter_values_dict
            and "x1_treshold" in parameter_values_dict
        ):
            alpha = jnp.where(
                self._data["x1"] < parameter_values_dict["x1_treshold"],
                parameter_values_dict["alpha_low"],
                parameter_values_dict["alpha_high"],
            )
        else:
            alpha = parameter_values_dict["alpha"]

        observed_distance_modulus = (
            self._data["mb"]
            + alpha * self._data["x1"]
            - parameter_values_dict["beta"] * self._data["c"]
            - parameter_values_dict["M_0"]
        )
        if "p" in self._data and "gamma" in parameter_values_dict:
            observed_distance_modulus += (
                parameter_values_dict["gamma"] * self._data["p"]
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

            variance_distance_modulus = _variance_from_errors(
                self._data["e_mb"],
                self._data["e_x1"],
                self._data["e_c"],
                self._data["cov_mb_x1"],
                self._data["cov_mb_c"],
                self._data["cov_x1_c"],
                parameter_values_dict["alpha"],
                parameter_values_dict["beta"],
                parameter_values_dict["sigma_M"],
            )

            if self.optional_covariance_observed_distance_modulus is not None:
                variance_distance_modulus = (
                    jnp.diag(variance_distance_modulus)
                    + self.optional_covariance_observed_distance_modulus
                )
        else:
            variance_distance_modulus = _variance_from_covariance(
                self._covariance_observation,
                self._number_datapoints,
                parameter_values_dict["alpha"],
                parameter_values_dict["beta"],
                parameter_values_dict["sigma_M"],
            )
            if self.optional_covariance_observed_distance_modulus is not None:
                variance_distance_modulus += (
                    self.optional_covariance_observed_distance_modulus
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


class VelCandleStandardized(DataVector):
    _kind = "velocity"
    _needed_keys = ["zobs", "mb"]
    _free_par = ["M_0", "sigma_M"]
    _number_dimension_observation_covariance = 1
    _parameters_observation_covariance = ["mb"]

    @property
    def conditional_needed_keys(self):
        """Conditionally required SALT2 error and covariance fields.

        Returns:
            list[str]: Includes errors and covariances when covariance is absent.
        """
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["e_mb"]
        return cond_keys

    def __init__(
        self,
        data,
        h,
        covariance_observation=None,
        optional_covariance_observed_distance_modulus=None,
        velocity_estimator="full",
    ):
        """Initialize SN Ia velocity vector from SALT2 fits.

        Args:
            data (dict): Includes SALT2 parameters and cosmology fields.
            h (float): Little-h scaling for distances.
            covariance_observation (ndarray|None): Optional observation covariance.
            velocity_estimator (str): Estimator name.
            mass_step (float): Threshold for host mass step correction.

        Raises:
            ValueError: If covariance shape is not adapted
        """
        super().__init__(data, covariance_observation=covariance_observation)
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
        self.velocity_estimator = velocity_estimator
        self.h = h

    def compute_observed_distance_modulus(self, parameter_values_dict):
        """Compute observed distance modulus from SALT2 fit parameters.

        Args:
            parameter_values_dict (dict): Includes `alpha`, `beta`, `M_0`, optionally `gamma`.

        Returns:
            ndarray: Distance modulus per object.
        """

        observed_distance_modulus = self._data["mb"] - parameter_values_dict["M_0"]
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
                self._data["mb"] ** 2 + parameter_values_dict["sigma_M"] ** 2
            )
            if self.optional_covariance_observed_distance_modulus is not None:
                variance_distance_modulus = (
                    jnp.diag(variance_distance_modulus)
                    + self.optional_covariance_observed_distance_modulus
                )
        else:
            variance_distance_modulus = (
                self._covariance_observation
                + jnp.eye(self._number_datapoints)
                * parameter_values_dict["sigma_M"] ** 2
            )
            if self.optional_covariance_observed_distance_modulus is not None:
                variance_distance_modulus += (
                    self.optional_covariance_observed_distance_modulus
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


# Placeholder for backward compatibility, to be removed in future versions
VelFromSALTfit = VelTrippRelation
