import abc
from inspect import getmembers, isfunction

import numpy as np

import flip.utils as utils
from flip.covariance import CovMatrix
from flip.utils import create_log

from . import cosmo_utils

try:
    import jax.numpy as jnp
    import jax.scipy as jsc
    from jax import jit

    jax_installed = True
except ImportError:
    import numpy as jnp
    import scipy as jsc

    jax_installed = False

log = create_log()

_avail_velocity_estimator = ["watkins", "lowz", "hubblehighorder", "full"]


def redshift_dependence_velocity(data, velocity_estimator, **kwargs):
    prefactor = utils._C_LIGHT_KMS_ * jnp.log(10) / 5
    redshift_obs = jnp.array(data["zobs"])

    if velocity_estimator == "watkins":
        redshift_dependence = prefactor * redshift_obs / (1 + redshift_obs)
    elif velocity_estimator == "lowz":
        redshift_dependence = prefactor / ((1 + redshift_obs) / redshift_obs - 1.0)
    elif velocity_estimator == "hubblehighorder":
        if ("q0" not in kwargs) & ("j0" not in kwargs):
            raise ValueError(
                """ The "q0" and "j0" parameters are not present in the **kwargs"""
                f""" Please add it or choose a different velocity_estimator among {_avail_velocity_estimator}"""
            )
        q_0 = kwargs["q0"]
        j_0 = kwargs["j0"]
        redshift_mod = redshift_obs * (
            1
            + (1 / 2) * (1 - q_0) * redshift_obs
            - (1 / 6) * (1 - q_0 - 3 * q_0**2 + j_0) * redshift_obs**2
        )
        redshift_dependence = prefactor * redshift_mod / (1 + redshift_obs)

    elif velocity_estimator == "full":
        if ("hubble_norm" not in data) | ("rcom_zobs" not in data):
            raise ValueError(
                """ The "hubble_norm" (H(z)/h = 100 E(z)) or "rcom_zobs" (Dm(z)) fields are not present in the data"""
                f""" Please add it or choose a different velocity_estimator among {_avail_velocity_estimator}"""
            )

        redshift_dependence = prefactor / (
            (1 + redshift_obs)
            * utils._C_LIGHT_KMS_
            / (jnp.array(data["hubble_norm"]) * jnp.array(data["rcom_zobs"]))
            - 1.0
        )

    else:
        raise ValueError(
            f"""Please choose a velocity_estimator from salt fit among {_avail_velocity_estimator}"""
        )
    return redshift_dependence


class DataVector(abc.ABC):
    _free_par = []
    
    @property
    def free_par(self):
        return self._free_par

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
    def _give_data_and_errors(self, **kwargs):
        pass

    def _check_keys(self, data):
        for k in self.needed_keys:
            if k not in data:
                raise ValueError(f"{k} field is needed in data")

    def __init__(self, data, cov=None, **kwargs):
        self._cov = cov
        self._check_keys(data)
        self._data = data
        self._kwargs = kwargs

        for k in self._data:
            self._data[k] = jnp.array(self._data[k])

    def __call__(self, *args):
        return self._give_data_and_errors(*args)

    def mask(self, bool_mask):
        if len(bool_mask) != len(self.data[self.needed_keys[0]]):
            raise ValueError("Boolean mask does not align with data")
        new_data = {k: v[bool_mask] for k, v in self._data.items()}

        new_cov = None
        if self._cov is not None:
            new_cov = self._cov[np.outer(bool_mask, bool_mask)]
        return type(self)(new_data, cov=new_cov, **self._kwargs)

    def compute_cov(self, model, power_spectrum_dict, **kwargs):

        coords = np.vstack((self.data["ra"], self.data["dec"], self.data["rcom_zobs"]))

        return CovMatrix.init_from_flip(
            model,
            self._kind,
            power_spectrum_dict,
            **{f"coordinates_{self._kind}": coords},
            **kwargs,
        )


class Density(DataVector):
    _kind = "density"
    _needed_keys = ["density", "density_error"]

    def _give_data_and_errors(self, *args):
        return self._data["density"], self._data["density_error"]


class DirectVel(DataVector):
    _kind = "velocity"
    _needed_keys = ["velocity"]

    @property
    def conditional_needed_keys(self):
        cond_keys = []
        if self._cov is None:
            cond_keys += ["velocity_error"]
        return cond_keys

    def _give_data_and_errors(self, *args):
        if self._cov is not None:
            return self._data["velocity"], self._cov
        return self._data["velocity"], self._data["velocity_error"]


class DensVel(DataVector):
    _kind = "cross"

    @property
    def needed_keys(self):
        return self.densities.needed_keys + self.velocities.needed_keys

    @property
    def free_par(self):
        return self.densities.free_par + self.velocities.free_par

    def _give_data_and_errors(self, *args):
        data_dens, err_dens = self.densities._give_data_and_errors(*args)
        data_vel, err_vel = self.velocities._give_data_and_errors(*args)
        
        data = np.hstack((data_dens, data_vel))
        errors = np.hstack((err_dens, err_vel))
        return data, errors

    def __init__(self, DensityVector, VelocityVector):
        self.densities = DensityVector
        self.velocities = VelocityVector

        if self.velocities._cov is not None:
            raise NotImplementedError("Vel with cov + density not implemented yet")
        
    def compute_cov(self, model, power_spectrum_dict, **kwargs):

        coords_dens = np.vstack((self.densities.data["ra"], 
                                 self.densities.data["dec"], 
                                 self.densities.data["rcom_zobs"]))
        
        coords_vel = np.vstack((self.velocities.data["ra"], 
                                 self.velocities.data["dec"], 
                                 self.velocities.data["rcom_zobs"]))
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
        if self._cov is None:
            cond_keys += ["dmu_error"]
        return self._needed_keys + cond_keys

    def _init_dmu2vel(self, vel_estimator, **kwargs):
        return redshift_dependence_velocity(self._data, vel_estimator, **kwargs)

    def __init__(self, data, cov=None, vel_estimator="full", **kwargs):
        super().__init__(data, cov=cov)
        self._dmu2vel = self._init_dmu2vel(vel_estimator, **kwargs)
        self._data["velocity"] = self._dmu2vel * self._data["dmu"]

        if self._cov is not None:
            self._cov = self._dmu2vel.T @ self._cov @ self._dmu2vel
        else:
            self._data["velocity_error"] = self._dmu2vel * self._data["dmu_error"]
