import numpy as np

from flip import utils

log = utils.create_log()


def compute_sep(
    ra,
    dec,
    comoving_distance,
    los_definition="bisector",
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
        r, theta, _ = angle_separation(
            ra_i, ra_j, dec_i, dec_j, r_i, r_j, los_definition=los_definition
        )
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


def compute_phi(ra_0, ra_1, dec_0, dec_1, r_0, r_1, los_definition):
    """
    The compute_phi function computes the angle between the line of sight and
    the separation vector. The line of sight is defined as either:
        - mean: (r_0 + r_2) / 2, where r_0 and r_2 are the radial coordinates at each end of a pair.
        - bisector: (r_0 / |r| + r_2 / |r|), where |r| is the distance between two points in a pair.
        - endpoint: only use one point in a pair to define the line-of-sight direction, i.e., use only one galaxy

    Args:
        ra_0: Compute the x_0, y_0 and z_0 coordinates of a galaxy
        ra_1: Compute the phi angle
        dec_0: Compute the z_0 parameter in radec2cart function
        dec_1: Compute the phi angle
        r_0: Compute the distance between two points
        r_1: Compute the distance between two points in the sky
        los_definition: Define the line of sight

    Returns:
        The angle between the line of sight and the separation vector
    """
    x_0, y_0, z_0 = utils.radec2cart(r_0, ra_0, dec_0)
    x_1, y_1, z_1 = utils.radec2cart(r_1, ra_1, dec_1)

    r_x = x_1 - x_0
    r_y = y_1 - y_0
    r_z = z_1 - z_0

    # We chose here r = r1 - r0, this could be the other way.
    # It should not change the maximum likelihood minimization.
    # CR - Need to be verified at one point. It could only impact vg.

    r = np.sqrt(r_x**2 + r_y**2 + r_z**2)
    if los_definition == "mean":
        d_x = x_0 + x_1
        d_y = y_0 + y_1
        d_z = z_0 + z_1
    elif los_definition == "bisector":
        d_x = x_0 / r_0 + x_1 / r_1
        d_y = y_0 / r_0 + y_1 / r_1
        d_z = z_0 / r_0 + z_1 / r_1
    elif los_definition == "endpoint":
        d_x = x_0
        d_y = y_0
        d_z = z_0
    else:
        raise ValueError(
            "Please choose a correlation_method between bisector, mean or endpoint"
            "Bisector method advised, endpoint not recommended"
        )
    d = np.sqrt(d_x**2 + d_y**2 + d_z**2)

    mask = d * r == 0.0
    cos_phi = np.zeros_like(r)
    cos_phi[~mask] = (
        d_x[~mask] * r_x[~mask] + d_y[~mask] * r_y[~mask] + d_z[~mask] * r_z[~mask]
    ) / (d[~mask] * r[~mask])

    phi = np.arccos(np.clip(cos_phi, -1, 1))

    return phi


def angle_separation(
    ra_0,
    ra_1,
    dec_0,
    dec_1,
    r_0,
    r_1,
    los_definition="bisector",
):
    """
    The angle_separation function computes the angle between two points on a sphere.

    Args:
        ra_0: Set the right ascension of the first galaxy
        ra_1: Calculate the angle between two points in the sky
        dec_0: Define the declination of the first galaxy
        dec_1: Calculate the cosine of theta
        r_0: Compute the distance between two points
        r_1: Compute the distance between two points in space
        los_definition: Define the line of sight
        : Define the line of sight

    Returns:
        The separation between two points in
    """

    cos_theta = np.cos(ra_1 - ra_0) * np.cos(dec_0) * np.cos(dec_1) + np.sin(
        dec_0
    ) * np.sin(dec_1)
    theta = np.arccos(np.clip(cos_theta, -1, 1))

    r = np.sqrt(r_0**2 + r_1**2 - 2 * r_0 * r_1 * np.clip(cos_theta, -1, 1))
    phi = compute_phi(ra_0, ra_1, dec_0, dec_1, r_0, r_1, los_definition)
    return r, theta, phi


def compute_phi_bisector_theorem(r, theta, r_0, r_1):
    sin_phi = ((r_0 + r_1) / r) * np.sin(theta / 2)
    return np.arcsin(np.clip(sin_phi, -1, 1))


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


def return_flat_cov(cov):
    variance_val = cov[0, 0]
    flat_cov = cov[np.triu_indices_from(cov, k=1)]
    flat_cov = np.insert(flat_cov, 0, variance_val)
    return flat_cov


def return_flat_cross_cov(cov):
    return cov.flatten()


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


def generator_need(
    coordinates_density=None,
    coordinates_velocity=None,
):
    """
    The generator_need function checks if the coordinates_density and coordinates_velocity inputs are provided.
    If they are not, it raises a ValueError exception.


    Args:
        coordinates_density: Generate the density covariance matrix
        coordinates_velocity: Generate the covariance matrix of the velocity field
        : Check if the coordinates are provided or not

    Returns:
        A list of the coordinates that are needed to proceed with covariance generation

    """
    if coordinates_density is not False:
        if coordinates_density is None:
            log.add(
                f"The coordinates_density input is needed to proceed covariance generation, please provide it"
            )
            raise ValueError("Density coordinates not provided")
    if coordinates_velocity is not False:
        if coordinates_velocity is None:
            log.add(
                f"The coordinates_velocity input is needed to proceed covariance generation, please provide it"
            )
            raise ValueError("Velocity coordinates not provided")


def check_generator_need(model_type, coordinates_density, coordinates_velocity):
    """
    The check_generator_need function is used to check if the generator_need function
    is called with the correct arguments. The model type determines which coordinates are needed,
    and these are passed as arguments to generator_need.

    Args:
        model_type: Determine if the density, velocity or full model is being used
        coordinates_density: Check if the density coordinates are needed
        coordinates_velocity: Determine whether the velocity model is needed

    Returns:
        A boolean

    """
    if model_type == "density":
        generator_need(
            coordinates_density=coordinates_density,
            coordinates_velocity=False,
        )
    if model_type == "velocity":
        generator_need(
            coordinates_density=False,
            coordinates_velocity=coordinates_velocity,
        )
    if model_type in ["full", "density_velocity"]:
        generator_need(
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
        )
