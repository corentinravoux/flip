import copy

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.sparse import coo_array

import flip.utils as utils

from .._config import __use_jax__

if __use_jax__:
    try:
        import jax.numpy as jnp

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:

    import numpy as jnp

    jax_installed = False

_avail_velocity_estimator = ["watkins", "lowz", "hubblehighorder", "full"]


def redshift_dependence_velocity(data, velocity_estimator, **kwargs):
    """Compute the redshift-dependent conversion from distance modulus to velocity.

    Args:
        data (dict): Must include `zobs` and, for `full`, `hubble_norm` and `rcom_zobs`.
        velocity_estimator (str): One of `"watkins"`, `"lowz"`, `"hubblehighorder"`, `"full"`, `"full_lcdm"`.
        **kwargs: Additional parameters required by specific estimators (e.g., `q0`, `j0`, `H0`, `Omega_m0`).

    Returns:
        ndarray: Conversion factor per object.

    Raises:
        ValueError: If required fields or parameters are missing or estimator unknown.
    """
    prefactor = -1.0 * utils._C_LIGHT_KMS_ * jnp.log(10) / 5
    redshift_obs = data["zobs"]

    if velocity_estimator == "watkins":
        redshift_dependence = prefactor * redshift_obs / (1 + redshift_obs)
    elif velocity_estimator == "lowz":
        redshift_dependence = prefactor * redshift_obs
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
            / (data["hubble_norm"] * data["rcom_zobs"])
            - 1.0
        )

    elif velocity_estimator == "full_lcdm":
        if ("H0" not in kwargs) & ("Omega_m0" not in kwargs):
            raise ValueError(
                """ The "H0" and "Omega_m0" parameters are not present in the **kwargs"""
                f""" Please add it or choose a different velocity_estimator among {_avail_velocity_estimator}"""
            )
        H0 = kwargs["H0"]
        Omega_m0 = kwargs["Omega_m0"]
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m0)

        rcom_zobs = cosmo.comoving_distance(redshift_obs).value
        hubble_z = cosmo.H(redshift_obs).value

        redshift_dependence = prefactor / (
            (1 + redshift_obs) * utils._C_LIGHT_KMS_ / (hubble_z * rcom_zobs) - 1.0
        )

    else:
        raise ValueError(
            f"""Please choose a velocity_estimator from salt fit among {_avail_velocity_estimator}"""
        )
    return redshift_dependence


def compute_host_matrix(host_group_id):
    """Build a sparse host-group assignment matrix and mapping.

    Args:
        host_group_id (array-like): Group id per object.

    Returns:
        tuple: `(coo_array, data_to_host_mapping)`.

    Raises:
        ValueError: If no grouping is present (all ids unique).
    """
    host_list, data_to_host_mapping = np.unique(host_group_id, return_inverse=True)
    if len(host_list) == len(host_group_id):
        raise ValueError(
            "len(host_list) == len(host_group_id), there is no group donot use host_group_id"
        )
    n_host = len(host_list)

    host_matrix = np.empty((n_host, len(host_group_id)), dtype=float)

    for i, h in enumerate(host_list):
        host_matrix[i] = host_group_id == h
    return coo_array(host_matrix), data_to_host_mapping


def format_data_multiple_host(data, sparse_host_matrix):
    """Aggregate per-host values and add `_full` copies of originals.

    For `ra`, use circular mean; for other supported fields, compute arithmetic
    means per host.

    Args:
        data (dict): Original per-object data.
        sparse_host_matrix (coo_array|BCOO): Host assignment matrix.

    Returns:
        dict: Aggregated data with host-level values.
    """
    data = data.copy()
    variable_to_mean = ["ra", "dec", "zobs", "rcom_zobs", "hubble_norm"]

    variable_to_process = (v for v in variable_to_mean if v in data)
    number_hosts = sparse_host_matrix.sum(axis=1)

    for v in variable_to_process:
        data[f"{v}_full"] = copy.copy(data[v])

        if v == "ra":
            # Circular mean for ra
            sinra = (sparse_host_matrix * data[v]).sin()
            cosra = (sparse_host_matrix * (np.pi / 2 - data[v])).sin()
            data[v] = np.arctan2(sinra.mean(axis=1), cosra.mean(axis=1))
            data[v] += 2 * np.pi * (data[v] < 0)
        else:
            data[v] = (sparse_host_matrix * data[v]).sum(axis=1) / number_hosts
    if jax_installed:
        for k in data:
            data[k] = jnp.array(data[k])
    return data


def get_grouped_data_variance(sparse_host_matrix, velocities, velocity_variance):
    """Compute grouped velocities and their propagated variance.

    Args:
        sparse_host_matrix (sparse matrix): Host assignment matrix.
        velocities (ndarray): Per-object velocities.
        velocity_variance (ndarray): Per-object variance or full covariance.

    Returns:
        tuple: `(group_velocities, group_variance_or_cov)`.
    """
    if velocity_variance.ndim == 1:
        weights = sparse_host_matrix / velocity_variance
    else:
        weights = sparse_host_matrix / jnp.diag(velocity_variance)

    if jax_installed:
        inverse_weigths_sum = 1 / weights.sum(axis=1).todense()
    else:
        inverse_weigths_sum = 1 / weights.sum(axis=1)

    velocities = weights @ velocities * inverse_weigths_sum

    if velocity_variance.ndim == 1:
        velocity_variance = inverse_weigths_sum
    else:
        weights = weights * inverse_weigths_sum[:, jnp.newaxis]
        velocity_variance = weights @ velocity_variance @ weights.T
    return velocities, velocity_variance
