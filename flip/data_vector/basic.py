import abc
import copy
import importlib

import numpy as np

from flip.covariance import CovMatrix
from flip.utils import create_log

from .._config import __use_jax__
from . import vector_utils

if __use_jax__:
    try:
        import jax.numpy as jnp
        from jax import jit, random
        from jax.experimental.sparse import BCOO

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:

    import numpy as jnp
    from numpy import random

    jax_installed = False

log = create_log()


class DataVector(abc.ABC):
    """Abstract base for data vectors used in fits.

    Provides common storage, key validation, optional JAX acceleration,
    covariance-aware masking, and covariance construction helpers.

    Attributes:
        _free_par (list[str]): Model parameters this vector depends on.
        _kind (str): One of "velocity", "density" or "cross".
    """

    _free_par = []
    _kind = ""  # 'velocity', 'density' or 'cross'

    @property
    def conditional_free_par(self):
        """Conditional extra parameters required by this vector.

        Returns:
            list[str]: Parameter names required depending on data content.
        """
        return []

    @property
    def free_par(self):
        """All free parameters for this vector.

        Returns:
            list[str]: Base plus conditional parameters.
        """
        return self._free_par + self.conditional_free_par

    @property
    def kind(self):
        """Return the data vector type.

        Returns:
            str: "velocity", "density" or "cross".
        """
        return self._kind

    @property
    def conditional_needed_keys(self):
        """Keys conditionally required in input `data`.

        Returns:
            list[str]: Extra keys required depending on configuration.
        """
        return []

    @property
    def needed_keys(self):
        """All required keys for this data vector.

        Returns:
            list[str]: Static plus conditional keys.
        """
        return self._needed_keys + self.conditional_needed_keys

    @property
    def data(self):
        """Access the underlying data dictionary.

        Returns:
            dict: Data fields as arrays.
        """
        return self._data

    @abc.abstractmethod
    def give_data_and_variance(self, **kwargs):
        """Return data vector and its variance/covariance.

        Returns:
            tuple: (data_array, variance_or_cov).
        """
        pass

    def _check_keys(self, data):
        """Validate that `data` contains all required keys.

        Raises:
            ValueError: When a required key is missing.
        """
        for k in self.needed_keys:
            if k not in data:
                raise ValueError(f"{k} field is needed in data")

    def __init__(self, data, covariance_observation=None, **kwargs):
        """Initialize data vector with data and optional observation covariance.

        Args:
            data (dict): Mapping of required fields to arrays.
            covariance_observation (ndarray|None): Observation covariance matrix or None.
            **kwargs: Extra configuration for subclasses.
        """
        self._covariance_observation = covariance_observation
        self._check_keys(data)
        self._data = copy.copy(data)
        self._kwargs = kwargs

        for k in self._data:
            self._data[k] = jnp.array(self._data[k])

        if jax_installed:
            self.give_data_and_variance_jit = jit(self.give_data_and_variance)

    def get_masked_data_and_cov(self, bool_mask):
        """Return masked data and corresponding masked observation covariance.

        Args:
            bool_mask (array-like): Boolean mask aligned with first data key length.

        Returns:
            tuple: (new_data_dict, new_cov) with covariance masked or None.

        Raises:
            ValueError: If mask length mismatches data length.
        """
        if len(bool_mask) != len(self.data[self.needed_keys[0]]):
            raise ValueError("Boolean mask does not align with data")
        new_data = {k: v[bool_mask] for k, v in self._data.items()}

        new_cov = None
        if self._covariance_observation is not None:
            new_cov = self._covariance_observation[np.ix_(bool_mask, bool_mask)]
        return new_data, new_cov

    def compute_covariance(self, model, power_spectrum_dict, **kwargs):
        """Build a `CovMatrix` for this vector and model.

        Args:
            model (str): Covariance model module under `flip.covariance`.
            power_spectrum_dict (dict): Power spectra inputs for model.
            **kwargs: Model-specific options.

        Returns:
            CovMatrix: Initialized covariance matrix object.
        """

        coordinate_keys = importlib.import_module(
            f"flip.covariance.analytical.{model}"
        )._coordinate_keys

        coords = np.vstack([self.data[k] for k in coordinate_keys])

        return CovMatrix.init_from_flip(
            model,
            self._kind,
            power_spectrum_dict,
            **{f"coordinates_{self._kind}": coords},
            **kwargs,
        )


class Dens(DataVector):
    _kind = "density"
    _needed_keys = ["density", "density_error"]

    def give_data_and_variance(self, *args):
        """Return density data and diagonal variance from `density_error`.

        Returns:
            tuple: (density, density_error^2).
        """
        return self._data["density"], self._data["density_error"] ** 2


class DirectVel(DataVector):
    _kind = "velocity"
    _needed_keys = ["velocity"]

    @property
    def conditional_needed_keys(self):
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["velocity_error"]
        return cond_keys

    def give_data_and_variance(self, *args):
        if self._covariance_observation is not None:
            return self._data["velocity"], self._covariance_observation
        return self._data["velocity"], self._data["velocity_error"] ** 2

    def __init__(self, data, covariance_observation=None):
        super().__init__(data, covariance_observation=covariance_observation)

        if "host_group_id" in self._data:
            # Copy full length velocities and velocity errors
            self._data["velocity_full"] = copy.copy(self._data["velocity"])

            # Init host matrix
            self._host_matrix, self._data_to_group_mapping = (
                vector_utils.compute_host_matrix(self._data["host_group_id"])
            )
            self._data = vector_utils.format_data_multiple_host(
                self._data, self._host_matrix
            )

            if jax_installed:
                self._host_matrix = BCOO.from_scipy_sparse(self._host_matrix)

            if self._covariance_observation is None:
                self._data["velocity_error_full"] = copy.copy(
                    self._data["velocity_error"]
                )
                velocity_variance = self._data["velocity_error"] ** 2
            else:
                velocity_variance = self._covariance_observation

            self._data["velocity"], velocity_variance = (
                vector_utils.get_grouped_data_variance(
                    self._host_matrix, self._data["velocity"], velocity_variance
                )
            )

            if self._covariance_observation is None:
                self._data["velocity_error"] = jnp.sqrt(velocity_variance)
            else:
                self._covariance_observation = velocity_variance


class DensVel(DataVector):
    _kind = "cross"

    @property
    def needed_keys(self):
        return self.densities.needed_keys + self.velocities.needed_keys

    @property
    def free_par(self):
        return self.densities.free_par + self.velocities.free_par

    def give_data_and_variance(self, *args):
        data_density, density_variance = self.densities.give_data_and_variance(*args)
        data_velocity, velocity_variance = self.velocities.give_data_and_variance(*args)
        data = jnp.hstack((data_density, data_velocity))
        variance = jnp.hstack((density_variance, velocity_variance))
        return data, variance

    def __init__(self, density_vector, velocity_vector):
        self.densities = density_vector
        self.velocities = velocity_vector

        if self.velocities._covariance_observation is not None:
            raise NotImplementedError(
                "Velocity with observed covariance + density not implemented yet"
            )

        if jax_installed:
            self.give_data_and_variance_jit = jit(self.give_data_and_variance)

    def compute_covariance(self, model, power_spectrum_dict, **kwargs):

        coords_dens = np.vstack(
            (
                self.densities.data["ra"],
                self.densities.data["dec"],
                self.densities.data["rcom_zobs"],
            )
        )

        coords_vel = np.vstack(
            (
                self.velocities.data["ra"],
                self.velocities.data["dec"],
                self.velocities.data["rcom_zobs"],
            )
        )
        return CovMatrix.init_from_flip(
            model,
            "full",
            power_spectrum_dict,
            coordinates_density=coords_dens,
            coordinates_velocity=coords_vel,
            **kwargs,
        )


class VelFromHDres(DataVector):
    _needed_keys = ["dmu", "zobs"]
    _free_par = ["M_0"]

    @property
    def conditional_needed_keys(self):
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["dmu_error"]
        return self._needed_keys + cond_keys

    def give_data_and_variance(self, parameter_values_dict):
        distance_modulus_difference_to_velocity = (
            vector_utils.redshift_dependence_velocity(
                self._data, self.velocity_estimator, **parameter_values_dict
            )
        )
        velocity = (
            distance_modulus_difference_to_velocity * self._data["dmu"]
            - distance_modulus_difference_to_velocity * parameter_values_dict["M_0"]
        )
        if self._covariance_observation is None and "dmu_error" in self._data:
            velocity_error = (
                distance_modulus_difference_to_velocity * self._data["dmu_error"]
            )
            return velocity, velocity_error**2

        elif self._covariance_observation is not None:
            J = jnp.diag(self._distance_modulus_difference_to_velocity)
            velocity_variance = J @ self._covariance_observation @ J.T
            return velocity, velocity_variance
        else:
            raise ValueError(
                "Cannot compute velocity variance without dmu_error or covariance_observation"
            )

    def __init__(
        self, data, covariance_observation=None, velocity_estimator="full", **kwargs
    ):
        # Compute conversion using provided input data, not uninitialized self._data

        self.velocity_estimator = velocity_estimator

        super().__init__(data, covariance_observation=covariance_observation)


class VelFromIntrinsicScatter(DataVector):
    _kind = "velocity"
    _needed_keys = ["zobs"]
    _free_par = ["sigma_M"]

    def give_data_and_variance(self, parameter_values_dict):
        distance_modulus_difference_to_velocity = (
            vector_utils.redshift_dependence_velocity(
                self._data, self.velocity_estimator, **parameter_values_dict
            )
        )
        if jax_installed:
            key = random.PRNGKey(0)
            distance_modulus = parameter_values_dict["sigma_M"] * random.normal(
                key, (len(self._data["zobs"]),)
            )
        else:
            distance_modulus = random.normal(
                loc=0.0,
                scale=parameter_values_dict["sigma_M"],
                size=len(self._data["zobs"]),
            )

        variance = parameter_values_dict["sigma_M"] ** 2

        return (
            distance_modulus_difference_to_velocity * distance_modulus,
            distance_modulus_difference_to_velocity**2 * variance,
        )

    def __init__(self, data, velocity_estimator="full"):
        super().__init__(data)
        self.velocity_estimator = velocity_estimator
