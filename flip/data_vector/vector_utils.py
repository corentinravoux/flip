import numpy as np
from scipy.sparse import coo_array

try:
    import jax.numpy as jnp
    jax_installed = True
except ImportError:
    import numpy as jnp
    jax_installed = False


def compute_host_matrix(host_group_id):
    host_list, host_list_index = np.unique(
            host_group_id, 
            return_index=True)

    n_host = len(host_list)
    
    host_matrix = np.empty(
        (n_host, len(host_group_id)), 
        dtype=float
        )

    for i, h in enumerate(host_list):
        host_matrix[i] = host_group_id == h
    return coo_array(host_matrix)

def format_data_multiple_host(data, sparse_host_matrix):
    data = data.copy()
    variable_to_mean = ['ra', 'dec', 'zobs']
    if 'rcom_zobs' in data:
        variable_to_mean.append('rcom_zobs')
    if 'hubble_norm' in data:
        variable_to_mean.append('hubble_norm')

    for v in variable_to_mean:
        data[f'{v}_full'] = data[v].copy()
        
        if v == 'ra':
            # Circular mean for ra
            sinra = (sparse_host_matrix * data[v]).sin()
            cosra = (sparse_host_matrix * (np.pi / 2 - data[v])).sin()
            data[v] = np.arctan2(sinra.mean(axis=1), cosra.mean(axis=1))
            data[v] += 2*np.pi * (data[v] < 0)
        else:
            data[v] = (sparse_host_matrix * data[v]).mean(axis=1)
    if jax_installed:
        for k in data:
            data[k] = jnp.array(data[k])
    return data