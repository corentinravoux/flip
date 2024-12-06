import abc
import copy
import importlib

import numpy as np

import flip.utils as utils
from flip.covariance import CovMatrix
from flip.utils import create_log

from . import vector_utils as vec_ut

try:
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    jax_installed = True
except ImportError:
    import numpy as jnp

    jax_installed = False

log = create_log()


class DataVector(abc.ABC):
    _free_par = []
    _kind = ""  # 'velocity', 'density' or 'cross'

    @property
    def conditional_free_par(self):
        return []

    @property
    def free_par(self):
        return self._free_par + self.conditional_free_par

    @property
    def kind(self):
        return self._kind

    @property
    def conditional_needed_keys(self):
        return []

    @property
    def needed_keys(self):
        return self._needed_keys + self.conditional_needed_keys

    @property
    def data(self):
        return self._data

    @abc.abstractmethod
    def _give_data_and_variance(self, **kwargs):
        pass

    def _check_keys(self, data):
        for k in self.needed_keys:
            if k not in data:
                raise ValueError(f"{k} field is needed in data")

    def __init__(self, data, covariance_observation=None, **kwargs):
        self._covariance_observation = covariance_observation
        self._check_keys(data)
        self._data = copy.copy(data)
        self._kwargs = kwargs

        for k in self._data:
            self._data[k] = jnp.array(self._data[k])

    def __call__(self, *args):
        return self._give_data_and_variance(*args)

    def mask(self, bool_mask):
        if len(bool_mask) != len(self.data[self.needed_keys[0]]):
            raise ValueError("Boolean mask does not align with data")
        new_data = {k: v[bool_mask] for k, v in self._data.items()}

        new_cov = None
        if self._covariance_observation is not None:
            new_cov = self._covariance_observation[np.ix_(bool_mask, bool_mask)]
        return type(self)(new_data, covariance_observation=new_cov, **self._kwargs)

    def compute_covariance(self, model, power_spectrum_dict, **kwargs):

        coordinate_keys = importlib.import_module(
            f"flip.covariance.{model}"
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

    def _give_data_and_variance(self, *args):
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

    def _give_data_and_variance(self, *args):
        if self._covariance_observation is not None:
            return self._data["velocity"], self._covariance_observation
        return self._data["velocity"], self._data["velocity_error"] ** 2

    def __init__(self, data, covariance_observation=None):
        super().__init__(data, covariance_observation=covariance_observation)

        if "host_group_id" in self._data:
            # Copy full length velocities and velocity errors
            self._data["velocity_full"] = copy.copy(self._data["velocity"])

            # Init host matrix
            self._host_matrix, self._data_to_group_mapping = vec_ut.compute_host_matrix(
                self._data["host_group_id"]
            )
            self._data = vec_ut.format_data_multiple_host(self._data, self._host_matrix)

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
                vec_ut.get_grouped_data_variance(
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

    def _give_data_and_variance(self, *args):
        data_density, density_variance = self.densities._give_data_and_variance(*args)
        data_velocity, velocity_variance = self.velocities._give_data_and_variance(
            *args
        )

        data = np.hstack((data_density, data_velocity))
        variance = np.hstack((density_variance, velocity_variance))
        return data, variance

    def __init__(self, density_vector, velocity_vector):
        self.densities = density_vector
        self.velocities = velocity_vector

        if self.velocities._covariance_observation is not None:
            raise NotImplementedError(
                "Velocity with observed covariance + density not implemented yet"
            )

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


class VelFromHDres(DirectVel):
    _needed_keys = ["dmu", "zobs"]

    @property
    def conditional_needed_keys(self):
        cond_keys = []
        if self._covariance_observation is None:
            cond_keys += ["dmu_error"]
        return self._needed_keys + cond_keys

    def __init__(
        self, data, covariance_observation=None, velocity_estimator="full", **kwargs
    ):

        self._distance_modulus_difference_to_velocity = (
            vec_ut.redshift_dependence_velocity(data, velocity_estimator, **kwargs)
        )

        data["velocity"] = self._distance_modulus_difference_to_velocity * data["dmu"]

        if covariance_observation is not None:
            J = jnp.diag(self._distance_modulus_difference_to_velocity)
            covariance_observation = J @ covariance_observation @ J.T
        else:
            data["velocity_error"] = (
                self._distance_modulus_difference_to_velocity * data["dmu_error"]
            )
        super().__init__(data, covariance_observation=covariance_observation)


class FisherVelFromHDres(DataVector):
    _kind = "velocity"
    _needed_keys = ["zobs", "ra", "dec", "rcom_zobs"]
    _free_par = ["sigma_M"]

    def _give_data_and_variance(self, parameter_values_dict):

        variance = parameter_values_dict["sigma_M"] ** 2
        if "dmu_error" in self.data:
            variance += self.data["dmu_error"] ** 2
        return self._distance_modulus_difference_to_velocity**2 * variance

    def __init__(self, data, velocity_estimator="full", **kwargs):
        super().__init__(data)
        self._distance_modulus_difference_to_velocity = (
            vec_ut.redshift_dependence_velocity(
                self._data, velocity_estimator, **kwargs
            )
        )


class FisherDens(DataVector):
    _kind = "density"
    _needed_keys = ["ra", "dec", "rcom_zobs"]
    _free_par = []

    def _give_data_and_variance(self, parameter_values_dict):
        variance = 0
        if "density_error" in self.data:
            variance += self.data["density_error"] ** 2
        return variance

    def __init__(self, data, velocity_estimator="full", **kwargs):
        super().__init__(data)


class FisherDensVel(DataVector):
    _kind = "cross"

    def _give_data_and_variance(self, *args):
        density_variance = self.densities._give_data_and_variance(*args)
        velocity_variance = self.velocities._give_data_and_variance(*args)

        variance = np.hstack((density_variance, velocity_variance))
        return variance

    def __init__(self, fisher_density, fisher_velocity):
        self.densities = fisher_density
        self.velocities = fisher_velocity

        if self.velocities._covariance_observation is not None:
            raise NotImplementedError("Vel with cov + density not implemented yet")

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
