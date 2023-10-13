import numpy as np


def compute_sep(
    ra,
    dec,
    comoving_distance,
    size_batch=10_000,
):
    """
    The compute_sep function computes the separation between all pairs of objects in a catalog.

    Args:
        ra: Store the right ascension of each object
        dec: Compute the angular separation between two objects
        comoving_distance: Calculate the separation between two objects
        size_batch: Control the number of objects that are processed at a time
        : Set the number of objects in a batch

    Returns:
        The separation, the perpendicular component of the separation, and

    """
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
    """
    The compute_i_j function takes in a number of nodes N and a sequence number seq.
    It returns the i, j indices for the upper triangular matrix that correspond to
    the given sequence number. The function is used to convert between an indexing scheme
    that uses integers from 0 to (N^2 - N)/2 - 1 and one that uses i,j indices where
    i = 0,...,N-2 and j = i+ 1,...,N-2.

    Args:
        N: Determine the size of the matrix
        seq: Find the row and column of the element in a matrix

    Returns:
        The row and column of the lower triangle matrix

    """
    i = (N - 2 - np.floor(np.sqrt(-8 * seq + 4 * N * (N - 1) - 7) / 2.0 - 0.5)).astype(
        "int"
    )
    j = (seq + i + 1 - N * (N - 1) / 2 + (N - i) * ((N - i) - 1) / 2).astype("int")
    return i, j


def compute_i_j_cross_matrix(Nv, seq):
    """
    The compute_i_j_cross_matrix function takes in the number of vertices, Nv, and a sequence of numbers
    and returns two arrays. The first array is an array containing the row indices for each element in the
    sequence. The second array contains column indices for each element in the sequence.

    Args:
        Nv: Determine the number of vertices in a given face
        seq: Create the i and j arrays

    Returns:
        The indices of the cross-terms in the matrix

    """
    i = np.floor(1 + (seq - Nv) / Nv).astype("int")
    j = (seq - i * Nv).astype("int")
    return i, j


def angle_separation(ra_0, ra_1, dec_0, dec_1, r_0, r_1):
    """
    The angle_separation function calculates the angle between two points on a sphere.

    Args:
        ra_0: Calculate the cosine of the angle between two points in spherical coordinates
        ra_1: Calculate the cosine of the angle between two points
        dec_0: Calculate the cosine of the angle between two points
        dec_1: Calculate the cos_theta parameter
        r_0: Calculate the distance between two points
        r_1: Calculate the distance between two objects

    Returns:
        The separation angle between two points

    """
    cos_theta = np.cos(ra_1 - ra_0) * np.cos(dec_0) * np.cos(dec_1) + np.sin(
        dec_0
    ) * np.sin(dec_1)
    r = np.sqrt(r_0**2 + r_1**2 - 2 * r_0 * r_1 * cos_theta)
    sin_phi = ((r_0 + r_1) / r) * np.sqrt((1 - cos_theta) / 2)
    sin_phi = np.clip(sin_phi, -1, 1)
    cos_theta = np.clip(cos_theta, -1, 1)
    return r, np.arccos(cos_theta), np.arcsin(sin_phi)


def return_full_cov(cov):
    """
    The return_full_cov function takes in a 1D array of covariance values and returns the full covariance matrix.
    The first value in the input array is assumed to be the variance, and all other values are assumed to be non-diagonal
    covariance terms. The function then creates an empty square matrix with dimensions equal to the number of objects
    (calculated from cov_array size), fills it with zeros, adds diagonal elements (variance) using numpy's eye function,
    and finally fills in upper triangle elements (non-diagonal covariances) using numpy's triu_indices function

    Args:
        cov: Store the covariance matrix of a multivariate normal distribution

    Returns:
        A full covariance matrix from a vector of covariances

    """
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
    """
    The return_full_cov_cross function takes in a covariance matrix and the number of objects in each band.
    It then reshapes the covariance matrix into a full cross-covariance matrix, which is returned.

    Args:
        cov: Reshape the covariance matrix
        number_objects_g: Reshape the covariance matrix into a full covariance matrix
        number_objects_v: Reshape the covariance matrix into a full covariance matrix

    Returns:
        The full covariance matrix

    """
    full_cov = cov.reshape((number_objects_g, number_objects_v))
    return full_cov


def return_correlation_matrix(cov):
    """
    The return_correlation_matrix function takes a covariance matrix as input and returns the correlation matrix.
    The correlation matrix is calculated by dividing each element of the covariance matrix by the product of its row's standard deviation and column's standard deviation.

    Args:
        cov: Calculate the correlation matrix

    Returns:
        The correlation matrix

    """
    sigma = np.sqrt(np.diag(cov))
    corr_matrix = cov / np.outer(sigma, sigma)
    return corr_matrix


def save_matrix(matrix, name):
    """
    The save_matrix function takes a matrix and saves it to the current directory as a .npy file.

    Args:
        matrix: Save the matrix to a file
        name: Save the matrix with a name

    Returns:
        The name of the file that was saved

    """
    np.save(f"{name}.npy", matrix)


def open_matrix(name):
    """
    The open_matrix function takes in a string as an argument and returns the matrix that is saved under that name.

    Args:
        name: Specify the name of the matrix to be loaded

    Returns:
        A matrix

    """
    matrix = np.load(f"{name}.npy")
    return matrix
