import numpy as np
import copy
from scipy.sparse import coo_array

try:
    import jax.numpy as jnp

    jax_installed = True
except ImportError:
    import numpy as jnp

    jax_installed = False


def compute_host_matrix(host_group_id):
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
