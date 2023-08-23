import numpy as np


def compute_sep(
    ra,
    dec,
    comoving_distance,
    size_batch=10_000,
):
    number_objects = len(ra)
    n_task = int((number_objects * (number_objects + 1)) / 2) - number_objects

    sep = []
    sep_perp = []
    sep_par = []
    for n in range(0, n_task, size_batch):
        batches = np.arange(n, np.min((n + size_batch, n_task)))
        i_list, j_list = compute_i_j(number_objects, batches)
        r_i, ra_i, dec_i = comoving_distance[i_list], ra[i_list], dec[i_list]
        r_j, ra_j, dec_j = comoving_distance[j_list], ra[j_list], dec[j_list]
        r, theta, _ = angle_separation(ra_i, ra_j, dec_i, dec_j, r_i, r_j)
        sep.append(r)
        sep_perp.append(r * np.sin(theta))
        sep_par.append(r * np.cos(theta))

    sep = np.concatenate(sep)
    sep_perp = np.concatenate(sep_perp)
    sep_par = np.concatenate(sep_par)

    sep = np.insert(sep, 0, 0)
    sep_perp = np.insert(sep_perp, 0, 0)
    sep_par = np.insert(sep_par, 0, 0)

    return sep, sep_perp, sep_par


def compute_i_j(N, seq):
    i = (N - 2 - np.floor(np.sqrt(-8 * seq + 4 * N * (N - 1) - 7) / 2.0 - 0.5)).astype(
        "int"
    )
    j = (seq + i + 1 - N * (N - 1) / 2 + (N - i) * ((N - i) - 1) / 2).astype("int")
    return i, j


def compute_i_j_cross_matrix(Nv, seq):
    i = np.floor(1 + (seq - Nv) / Nv).astype("int")
    j = (seq - i * Nv).astype("int")
    return i, j


def angle_separation(ra_0, ra_1, dec_0, dec_1, r_0, r_1):
    cos_theta = np.cos(ra_1 - ra_0) * np.cos(dec_0) * np.cos(dec_1) + np.sin(
        dec_0
    ) * np.sin(dec_1)
    r = np.sqrt(r_0**2 + r_1**2 - 2 * r_0 * r_1 * cos_theta)
    sin_phi = ((r_0 + r_1) / r) * np.sqrt((1 - cos_theta) / 2)
    sin_phi = np.clip(sin_phi, -1, 1)
    cos_theta = np.clip(cos_theta, -1, 1)
    return r, np.arccos(cos_theta), np.arcsin(sin_phi)


def return_full_cov(cov):
    variance_val = cov[0]

    non_diagonal_cov = np.delete(cov, 0)
    number_objects = int((1 + np.sqrt(1 + 8 * non_diagonal_cov.size)) / 2)

    variance_val = variance_val * np.eye(number_objects)

    full_cov = np.zeros((number_objects, number_objects))
    vi, vj = np.triu_indices(number_objects, k=1)
    full_cov[vi, vj] = non_diagonal_cov
    full_cov[vj, vi] = non_diagonal_cov

    full_cov = full_cov + variance_val
    return full_cov


def return_full_cov_cross(cov, number_objects_g, number_objects_v):
    full_cov = cov.reshape((number_objects_g, number_objects_v))
    return full_cov


def return_correlation_matrix(cov):
    sigma = np.sqrt(np.diag(cov))
    corr_matrix = cov / np.outer(sigma, sigma)
    return corr_matrix


def save_matrix(matrix, name):
    np.save(f"{name}.npy", matrix)


def open_matrix(name):
    matrix = np.load(f"{name}.npy")
    return matrix
