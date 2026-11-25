import multiprocessing as mp
import time
from functools import partial

import numpy as np

from flip._config import __use_jax__

if __use_jax__:
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:
    import numpy as jnp

    jax_installed = False

from flip import utils
from flip.data_vector.mesh_utils import paint_catalog
from flip.data_vector.randoms import (  # noqa: F401  (re-exported)
    _sky_kernels_to_cartesian,
    compute_random_counts_analytic,
    define_randoms,
    move_kernels,
)

_grid_kind_avail = ["ngp", "cic", "tsc", "pcs"]

_grid_order_dict = {
    "ngp": 1,
    "cic": 2,
    "tsc": 3,
    "pcs": 4,
}


def _compute_grid_window(grid_size, k, order, n):
    """Numerically compute isotropic grid assignment window.

    Uses spherical averaging over directions to produce a 1D window for
    resampler order (NGP/CIC/TSC/PCS) given grid size.

    Args:
        grid_size (float): Grid cell size.
        k (array-like): Wavenumbers at which to evaluate the window.
        order (int): Assignment order (1:NGP, 2:CIC, 3:TSC, 4:PCS).
        n (int): Number of angular samples for spherical averaging.

    Returns:
        numpy.ndarray: Window values for each `k`.
    """
    theta = jnp.linspace(0, jnp.pi, n)
    phi = jnp.linspace(0, 2 * jnp.pi, n)
    kx = jnp.outer(jnp.sin(theta), jnp.cos(phi))
    ky = jnp.outer(jnp.sin(theta), jnp.sin(phi))
    kz = jnp.outer(jnp.cos(theta), jnp.ones(n))

    # Forgotten in Howlett et al. formula
    # we add spherical coordinate solid angle element
    dthetaphi = jnp.outer(jnp.sin(theta), jnp.ones(phi.size))
    window = []
    for i in range(k.size):
        # the factor here has an extra jnp.pi because of the definition of jnp.sinc
        fact = (k[i] * grid_size) / (2 * jnp.pi)
        func = (
            jnp.sinc(fact * kx) * jnp.sinc(fact * ky) * jnp.sinc(fact * kz)
        ) ** order * dthetaphi
        win_theta = jnp.trapezoid(func, x=phi)
        window.append(jnp.trapezoid(win_theta, x=theta))
    window = jnp.asarray(window, dtype=jnp.result_type(k, float))
    window *= 1 / (4 * jnp.pi)
    return window


def compute_grid_window(grid_size, kh, kind="ngp", n=1000):
    """Compute grid assignment window for a given resampler kind.

    Args:
        grid_size (float): Grid cell size.
        kh (array-like): Wavenumbers at which to evaluate the window.
        kind (str): One of `ngp`, `ngp_errw`, `cic`, `tsc`, `pcs`.
        n (int): Angular samples for spherical averaging.

    Returns:
        numpy.ndarray|None: Window values, or None if `grid_size==0`.
    """

    if kind not in _grid_kind_avail:
        allowed = ", ".join(_grid_kind_avail)
        raise ValueError(f"INVALID GRID TYPE! Allowed kinds: {allowed}")
    if grid_size == 0:
        return None
    return _compute_grid_window(grid_size, kh, _grid_order_dict[kind], n)


def ngp_weight(ds):
    """Return nearest-grid-point assignment weights.

    Args:
        ds (array-like): Distance to the cell center in cell-size units.

    Returns:
        numpy.ndarray: NGP weights evaluated at `ds`.
    """
    abs_ds = jnp.abs(ds)
    w = 1.0 * (abs_ds < 1 / 2)
    return w


def ngp_errw_weight(ds):
    """Return NGP weights for the error-weighted assignment scheme.

    Args:
        ds (array-like): Distance to the cell center in cell-size units.

    Returns:
        numpy.ndarray: Assignment weights evaluated at `ds`.
    """
    return ngp_weight(ds)


def cic_weight(ds):
    """Return cloud-in-cell assignment weights.

    Args:
        ds (array-like): Distance to the cell center in cell-size units.

    Returns:
        numpy.ndarray: CIC weights evaluated at `ds`.
    """
    abs_ds = jnp.abs(ds)
    w = (1.0 - abs_ds) * (abs_ds < 1)
    return w


def tsc_weight(ds):
    """Return triangular-shaped-cloud assignment weights.

    Args:
        ds (array-like): Distance to the cell center in cell-size units.

    Returns:
        numpy.ndarray: TSC weights evaluated at `ds`.
    """
    abs_ds = jnp.abs(ds)
    w = (3 / 4 - ds**2) * (abs_ds < 1 / 2)
    w += 1 / 2 * (3 / 2 - abs_ds) ** 2 * ((1 / 2 <= abs_ds) & (abs_ds < 3 / 2))
    return w


def pcs_weight(ds):
    """Return piecewise-cubic-spline assignment weights.

    Args:
        ds (array-like): Distance to the cell center in cell-size units.

    Returns:
        numpy.ndarray: PCS weights evaluated at `ds`.
    """
    abs_ds = jnp.abs(ds)
    w = 1 / 6 * (4 - 6 * ds**2 + 3 * abs_ds**3) * (abs_ds < 1)
    w += 1 / 6 * (2 - abs_ds) ** 3 * ((1 <= abs_ds) & (abs_ds < 2))
    return w


def _get_mesh_attrs(
    boxsize,
    cellsize,
):
    """Derive mesh dimensions and box metadata from target scales.

    Args:
        boxsize (float): Target box size along each Cartesian axis.
        cellsize (float): Desired cell size along each Cartesian axis.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Mesh shape,
        adjusted box size, and box center.
    """
    boxcenter = 0.0
    boxsize = np.full((3,), boxsize, dtype="f8")

    cellsize = np.full((3,), cellsize, dtype="f8")
    nmesh = boxsize / cellsize
    nmesh = np.ceil(nmesh).astype("i8")
    nmesh += nmesh % 2  # to make it even
    boxsize = nmesh * cellsize  # enforce exact cellsize

    nmesh = np.full((3,), nmesh, dtype="i4")
    boxcenter = np.full((3,), boxcenter, dtype="f8")
    return nmesh, boxsize, boxcenter


def create_mesh(
    positions,
    boxsize,
    cellsize,
    assignement="ngp",
    weights=None,
    scaling=None,
):
    """Paint weighted positions onto a mesh.

    Args:
        positions (array-like): Cartesian positions with shape `(N, 3)`.
        boxsize (float): Box size used to create the mesh.
        cellsize (float): Target cell size.
        assignement (str): Assignment scheme among `ngp`, `ngp_errw`, `cic`,
            `tsc`, and `pcs`.
        weights (array-like, optional): Per-object weights to paint on the mesh.
        scaling (float, optional): Extra multiplicative factor applied to weights.

    Returns:
        flip.data_vector.mesh_utils.RealField: Painted real-space mesh.

    Raises:
        ValueError: If positions or weights contain non-finite values.
    """

    conversions = {
        "ngp": "nnb",
        "ngp_errw": "nnb",
        "cic": "cic",
        "tsc": "tsc",
        "pcs": "pcs",
    }
    resampler = conversions[assignement]

    nmesh, boxsize, boxcenter = _get_mesh_attrs(
        boxsize,
        cellsize,
    )

    return paint_catalog(
        positions,
        boxsize,
        nmesh,
        boxcenter=boxcenter,
        resampler=resampler,
        weights=weights,
        scaling=scaling,
    )


def cut_grid(
    grid,
    remove_nan_density=True,
    remove_empty_cells=False,
    n_cut=None,
    weight_min=None,
    rcom_max=None,
    xmax=None,
    ymax=None,
    zmax=None,
    remove_origin=False,
):
    """Apply selection cuts to a grid in-place.

    Args:
        grid (dict): Grid dictionary to modify.
        remove_nan_density (bool): Drop NaN density and errors.
        remove_empty_cells (bool): Drop cells with zero counts.
        n_cut (int, optional): Minimum objects per cell.
        weight_min (float, optional): Minimum weight per cell.
        rcom_max (float, optional): Radial cutoff.
        xmax (float, optional): X cutoff.
        ymax (float, optional): Y cutoff.
        zmax (float, optional): Z cutoff.
        remove_origin (bool): Remove the origin cell.

    Returns:
        None: Modifies `grid` in-place.
    """
    mask = jnp.full(grid["ra"].shape, True)
    if n_cut is not None:
        mask &= grid["nincell"] > n_cut
    if weight_min is not None:
        mask &= grid["sum_weights"] > weight_min
    if rcom_max is not None:
        mask &= grid["rcom_zobs"] < rcom_max
    if xmax is not None:
        mask &= jnp.abs(grid["x"]) < xmax
    if ymax is not None:
        mask &= jnp.abs(grid["y"]) < ymax
    if zmax is not None:
        mask &= jnp.abs(grid["z"]) < zmax
    if remove_nan_density:
        if "density" in grid:
            mask &= ~(jnp.isnan(grid["density"]))
        if "density_error" in grid:
            mask &= ~(jnp.isnan(grid["density_error"]))
    if remove_empty_cells:
        if "N_in_cell" in grid:
            mask &= ~(grid["N_in_cell"].astype(int) == 0)
    if remove_origin:
        mask = mask & ((grid["x"] != 0.0) | (grid["y"] != 0.0) | (grid["z"] != 0.0))
    for field in grid:
        grid[field] = grid[field][mask]


def cut_grid_type(
    grid,
    grid_type,
    rcom_max,
):
    if grid_type == "rect":
        cut_grid(
            grid,
            remove_nan_density=True,
            xmax=rcom_max,
            ymax=rcom_max,
            zmax=rcom_max,
            remove_origin=True,
        )

    if grid_type == "sphere":
        cut_grid(
            grid,
            remove_nan_density=True,
            rcom_max=rcom_max,
            remove_origin=True,
        )


def prepare_data_position_kernel(
    data_position_sky_kernel,
    rcom_max,
    overhead,
    random_method=None,
    Nrandom=None,
    coord_randoms=None,
    seed=None,
    d_rcom=10.0,
):

    data_position_kernel = _sky_kernels_to_cartesian(data_position_sky_kernel)

    if random_method is not None:
        xobj_random, yobj_random, zobj_random, _, kernel_randoms = define_randoms(
            random_method,
            Nrandom=Nrandom,
            coord_randoms=coord_randoms,
            max_coordinates=rcom_max + overhead,
            seed=seed,
            data_position_sky_kernel=data_position_sky_kernel,
            d_rcom=d_rcom,
        )

        randoms_positions = jnp.array([xobj_random, yobj_random, zobj_random]).T
    else:
        kernel_randoms = None
        randoms_positions = None

    return data_position_kernel, randoms_positions, kernel_randoms


def prepare_data_position(
    data_position_sky,
    rcom_max,
    overhead,
    random_method=None,
    Nrandom=None,
    coord_randoms=None,
    seed=None,
    data_position_sky_bandwidth=None,
):
    """Convert sky coordinates to Cartesian positions and prepare randoms.

    Args:
        data_position_sky (array-like): Sky coordinates with columns `(ra, dec, rcom)`.
        rcom_max (float): Maximum comoving radius of the retained region.
        overhead (float): Padding added to the mesh extent.
        random_method (str, optional): Random generation method passed to
            `define_randoms`.
        Nrandom (int, optional): Number of random points per object.
        coord_randoms (tuple, optional): Input random sky coordinates for the
            `file` random method.
        seed (int, optional): Random seed used when generating randoms.
        data_position_sky_bandwidth (array-like, optional): Per-object bandwidth
            matrices expressed in sky coordinates.

    Returns:
        tuple: Cartesian data positions, transformed bandwidth matrices or `None`,
        and random positions or `None`.
    """

    raobj = data_position_sky[:, 0]
    decobj = data_position_sky[:, 1]
    rcomobj = data_position_sky[:, 2]
    xobj, yobj, zobj = utils.spherical_to_cartesian(raobj, decobj, rcomobj)
    mask = jnp.abs(xobj) < rcom_max + overhead
    mask &= jnp.abs(yobj) < rcom_max + overhead
    mask &= jnp.abs(zobj) < rcom_max + overhead
    xobj, yobj, zobj = xobj[mask], yobj[mask], zobj[mask]

    if data_position_sky_bandwidth is not None:

        jacobian = utils.spherical_to_cartesian_jacobian(
            raobj[mask], decobj[mask], rcomobj[mask]
        )

        data_position_sky_bandwidth = data_position_sky_bandwidth[mask, :, :]

        data_position_bandwith = (
            jacobian.T
            @ data_position_sky_bandwidth
            @ jnp.transpose(jacobian.T, (0, 2, 1))
        )
    else:
        data_position_bandwith = None

    raobj, decobj, rcomobj = raobj[mask], decobj[mask], rcomobj[mask]
    if random_method is not None:
        xobj_random, yobj_random, zobj_random, mask_random, _ = define_randoms(
            random_method,
            xobj,
            yobj,
            zobj,
            raobj,
            decobj,
            rcomobj,
            Nrandom=Nrandom,
            coord_randoms=coord_randoms,
            max_coordinates=rcom_max + overhead,
            seed=seed,
        )
        randoms_positions = jnp.array([xobj_random, yobj_random, zobj_random]).T
    else:
        randoms_positions = None
        mask_random = None

    data_positions = jnp.array([xobj, yobj, zobj]).T

    return (
        data_positions,
        data_position_bandwith,
        randoms_positions,
        mask,
        mask_random,
    )


def define_grid_from_mesh(mesh_data, grid_size):
    """Extract voxel-center coordinates and sky coordinates from a mesh.

    Args:
        mesh_data: `RealField` used to define the grid geometry.
        grid_size (float): Cell size used to shift coordinates to voxel centers.

    Returns:
        tuple[numpy.ndarray, ...]: Cartesian and sky coordinates of all grid cells.
    """
    coord_mesh = jnp.array(
        jnp.meshgrid(
            jnp.sort(mesh_data.x[0][:, 0, 0]),
            jnp.sort(mesh_data.x[1][0, :, 0]),
            jnp.sort(mesh_data.x[2][0, 0, :]),
            indexing="ij",
        )
    )
    xgrid = jnp.ravel(coord_mesh[0, :, :, :])
    ygrid = jnp.ravel(coord_mesh[1, :, :, :])
    zgrid = jnp.ravel(coord_mesh[2, :, :, :])

    ragrid, decgrid, rcomgrid = utils.cartesian_to_spherical(xgrid, ygrid, zgrid)

    return xgrid, ygrid, zgrid, ragrid, decgrid, rcomgrid


def grid_data_density(
    data_position_sky,
    rcom_max,
    grid_size,
    grid_type,
    kind,
    Nrandom=10,
    random_method="cartesian",
    coord_randoms=None,
    min_count_random=0,
    overhead=20,
    seed=None,
    data_weights=None,
    randoms_weights=None,
    d_rcom=10.0,
    n_bins_sin_dec=1,
):
    """Grid data with pypower and compute density contrast on a mesh.

    Args:
        raobj (array-like): Right ascensions.
        decobj (array-like): Declinations.
        rcomobj (array-like): Comoving distances.
        rcom_max (float): Outer cutoff for grid.
        grid_size (float): Cell size.
        grid_type (str): `rect` or `sphere` cut behavior.
        kind (str): Resampler passed to CatalogMesh.
        Nrandom (int): Randoms per data object.
        random_method (str): A `define_randoms` method, or `analytic` for
            expected counts without randoms (`compute_random_counts_analytic`).
        interlacing (int): Interlacing factor.
        compensate (bool): Apply resampler compensation.
        coord_randoms (tuple, optional): Randoms coordinates for `file` method.
        min_count_random (float): Minimum random count (expected count for
            `analytic`) for valid error.
        overhead (float): Extra margin around cutoff.
        data_weights (array-like): Weights for the galaxy.
        randoms_weights (array-like): Weights for the random.
        d_rcom (float): rcom bin width (`analytic`); a few `grid_size`.
        n_bins_sin_dec (int): Number of sin(dec) bins (`analytic`).

    Returns:
        dict: Grid with positions, density contrast, errors, and counts.
    """

    kind = kind.lower()
    if kind not in _grid_kind_avail:
        allowed = ", ".join(_grid_kind_avail)
        raise ValueError(f"INVALID GRID TYPE! Allowed kinds: {allowed}")

    data_positions, _, randoms_positions, mask_data, mask_random = (
        prepare_data_position(
            data_position_sky,
            rcom_max,
            overhead,
            random_method=None if random_method == "analytic" else random_method,
            Nrandom=Nrandom,
            coord_randoms=coord_randoms,
            seed=seed,
        )
    )
    if data_weights is None:
        data_weights = jnp.ones((data_positions.shape[0],))
    else:
        data_weights = data_weights[mask_data]

    mesh_data = create_mesh(
        data_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=data_weights,
    )

    xgrid, ygrid, zgrid, ragrid, decgrid, rcomgrid = define_grid_from_mesh(
        mesh_data,
        grid_size,
    )

    if random_method == "analytic":
        raobj, decobj, rcomobj = jnp.asarray(data_position_sky)[mask_data].T
        count_randoms_value = compute_random_counts_analytic(
            raobj,
            decobj,
            rcomobj,
            data_weights,
            xgrid,
            ygrid,
            zgrid,
            grid_size,
            rcom_max + overhead,
            d_rcom=d_rcom,
            n_bins_sin_dec=n_bins_sin_dec,
        )
        count_randoms_value = jnp.asarray(
            count_randoms_value.reshape(mesh_data.value.shape)
        )
        randoms_value = count_randoms_value
        scaling = 1.0

    else:
        if randoms_weights is None:
            randoms_weights = jnp.ones((randoms_positions.shape[0],))
        else:
            randoms_weights = randoms_weights[mask_random]
        scaling = jnp.sum(data_weights) / jnp.sum(randoms_weights)

        randoms_value = create_mesh(
            randoms_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement=kind,
            weights=randoms_weights,
            scaling=scaling,
        ).value
        count_randoms_value = create_mesh(
            randoms_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement="ngp",
            weights=randoms_weights,
        ).value

    mask_randoms_nonzero = randoms_value != 0.0
    safe_randoms = jnp.where(mask_randoms_nonzero, randoms_value, 1.0)
    density_contrast = jnp.ravel(
        jnp.where(mask_randoms_nonzero, mesh_data.value / safe_randoms - 1, jnp.nan)
    )

    count_randoms = jnp.ravel(count_randoms_value)
    mask = count_randoms > min_count_random
    safe_count_randoms = jnp.where(mask, count_randoms, 1)
    density_contrast_err = jnp.where(
        mask, jnp.sqrt(1 / (safe_count_randoms * scaling)), jnp.nan
    )

    grid = {
        "x": xgrid,
        "y": ygrid,
        "z": zgrid,
        "ra": ragrid,
        "dec": decgrid,
        "rcom_zobs": rcomgrid,
        "density": density_contrast,
        "density_error": density_contrast_err,
        "value_data": jnp.ravel(mesh_data.value),
        "count_random": count_randoms,
        "count_random_scaled": count_randoms * scaling,
    }

    cut_grid_type(
        grid,
        grid_type,
        rcom_max,
    )

    return grid


def grid_data_density_multivariate_kernel(
    data_position_sky,
    data_position_sky_bandwidth,
    rcom_max,
    grid_size,
    grid_type,
    kind,
    Nrandom=10,
    random_method="cartesian",
    coord_randoms=None,
    min_count_random=0,
    overhead=20,
    seed=None,
    kernel="gaussian",
    cutoff_type=None,
    threshold=1e-5,
    data_weights=None,
    randoms_weights=None,
):
    """Grid density data after smoothing each object with a local kernel.

    Args:
        data_position_sky (array-like): Sky coordinates with columns `(ra, dec, rcom)`.
        data_position_sky_bandwidth (array-like): Per-object bandwidth matrices in
            sky coordinates.
        rcom_max (float): Outer cutoff for the final grid.
        grid_size (float): Cell size of the mesh.
        grid_type (str): Either `rect` or `sphere` for the final cut.
        kind (str): Assignment kernel among the supported mesh schemes.
        Nrandom (int): Number of random points per object.
        random_method (str): Strategy used to generate random catalogs.
        coord_randoms (tuple, optional): Input random sky coordinates for the
            `file` random method.
        min_count_random (int): Minimum random count required to define errors.
        overhead (float): Padding added to the mesh extent.
        seed (int, optional): Random seed used when generating randoms.
        kernel (str): Name of the multivariate kernel.
        cutoff_type (str, optional): Optional truncation criterion applied to the
            kernel weights.
        threshold (float): Threshold associated with `cutoff_type`.

    Returns:
        dict: Gridded density field, density errors, and grid coordinates.
    """

    kind = kind.lower()
    if kind not in _grid_kind_avail:
        allowed = ", ".join(_grid_kind_avail)
        raise ValueError(f"INVALID GRID TYPE! Allowed kinds: {allowed}")

    (
        data_positions,
        data_position_bandwith,
        randoms_positions,
        mask_data,
        mask_random,
    ) = prepare_data_position(
        data_position_sky,
        rcom_max,
        overhead,
        random_method=random_method,
        Nrandom=Nrandom,
        coord_randoms=coord_randoms,
        seed=seed,
        data_position_sky_bandwidth=data_position_sky_bandwidth,
    )

    if data_weights is None:
        data_weights = jnp.ones((data_positions.shape[0],))
    else:
        data_weights = data_weights[mask_data]
    if randoms_weights is None:
        randoms_weights = jnp.ones((randoms_positions.shape[0],))
    else:
        randoms_weights = randoms_weights[mask_random]

    # First mesh for grid definition
    mesh_data = create_mesh(
        data_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=data_weights,
    )

    xgrid, ygrid, zgrid, ragrid, decgrid, rcomgrid = define_grid_from_mesh(
        mesh_data,
        grid_size,
    )

    grid_positions = jnp.array([xgrid, ygrid, zgrid]).T

    # Every object paints its normalised kernel on the voxel centres, so the
    # objects can be summed per voxel before painting. Objects are processed
    # in chunks of identical shape (the last one padded with masked objects),
    # which bounds memory to chunk_size * number_of_voxels entries and lets
    # JAX compile each operation once.
    number_of_objects = data_positions.shape[0]
    number_of_voxels = grid_positions.shape[0]
    chunk_size = max(1, min(number_of_objects, 2**22 // number_of_voxels))
    number_of_chunks = -(-number_of_objects // chunk_size)
    number_of_padded = number_of_chunks * chunk_size - number_of_objects
    padded_positions = jnp.concatenate(
        [data_positions, jnp.zeros((number_of_padded, 3))], axis=0
    )
    padded_bandwidth = jnp.concatenate(
        [
            data_position_bandwith,
            jnp.broadcast_to(jnp.eye(3), (number_of_padded, 3, 3)),
        ],
        axis=0,
    )
    is_object = jnp.arange(number_of_chunks * chunk_size) < number_of_objects

    voxel_kernel_weights = jnp.zeros(number_of_voxels)
    for start in range(0, number_of_chunks * chunk_size, chunk_size):
        chunk = slice(start, start + chunk_size)
        kernel_weights = _multivariate_kernel_weights_chunk(
            padded_positions[chunk],
            padded_bandwidth[chunk],
            grid_positions,
            grid_size,
            kernel=kernel,
            cutoff_type=cutoff_type,
            threshold=threshold,
        )
        # Normalise each object to one; objects with no weight paint nothing.
        kernel_sum = jnp.sum(kernel_weights, axis=1, keepdims=True)
        keep = is_object[chunk][:, None] & (kernel_sum > 0)
        kernel_weights = jnp.where(
            keep, kernel_weights / jnp.where(keep, kernel_sum, 1.0), 0.0
        )
        voxel_kernel_weights = voxel_kernel_weights + jnp.sum(kernel_weights, axis=0)

    data_kernel_positions = grid_positions
    data_kernel_weights = (
        number_of_objects * voxel_kernel_weights / jnp.sum(voxel_kernel_weights)
    )

    mesh_data_kernel = create_mesh(
        data_kernel_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=data_kernel_weights,
    )
    scaling = jnp.sum(data_weights) / jnp.sum(randoms_weights)

    mesh_randoms = create_mesh(
        randoms_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=randoms_weights,
        scaling=scaling,
    )
    mesh_count_randoms = create_mesh(
        randoms_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement="ngp",
        weights=randoms_weights,
    )

    mask_randoms_nonzero = mesh_randoms.value != 0.0
    safe_randoms = jnp.where(mask_randoms_nonzero, mesh_randoms.value, 1.0)
    density_contrast = jnp.ravel(
        jnp.where(
            mask_randoms_nonzero, mesh_data_kernel.value / safe_randoms - 1, jnp.nan
        )
    )

    count_randoms = jnp.ravel(mesh_count_randoms.value)
    mask = count_randoms > min_count_random
    safe_count_randoms = jnp.where(mask, count_randoms, 1)
    density_contrast_err = jnp.where(
        mask, jnp.sqrt(1 / (safe_count_randoms * scaling)), jnp.nan
    )

    grid = {
        "x": xgrid,
        "y": ygrid,
        "z": zgrid,
        "ra": ragrid,
        "dec": decgrid,
        "rcom_zobs": rcomgrid,
        "density": density_contrast,
        "density_error": density_contrast_err,
        "value_data": jnp.ravel(mesh_data.value),
        "count_random": count_randoms,
        "count_random_scaled": count_randoms * scaling,
    }

    cut_grid_type(
        grid,
        grid_type,
        rcom_max,
    )

    return grid


def multivariate_kernel_density_estimation(
    data_position,
    bandwidth,
    grid_positions,
    grid_size,
    kernel="gaussian",
    cutoff_type=None,
    threshold=1e-5,
):
    """Evaluate multivariate kernel weights on a Cartesian grid.

    Args:
        data_position (array-like): Cartesian position of the object center.
        bandwidth (array-like): $3 \times 3$ covariance matrix of the kernel.
        grid_positions (array-like): Cartesian positions of voxel centers.
        grid_size (float): Cell size used to clip distances to voxel boundaries.
        kernel (str): Kernel family to evaluate.
        cutoff_type (str, optional): Optional truncation mode among
            `kernel_unnormalized`, `kernel`, or `distance`.
        threshold (float): Threshold used by `cutoff_type`.

    Returns:
        numpy.ndarray: Kernel weights evaluated at each grid position.

    Raises:
        ValueError: If the kernel or cutoff type is unsupported.
    """

    if kernel == "gaussian":

        distances_to_voxel = jnp.maximum(
            jnp.abs(data_position - grid_positions) - grid_size / 2, 0
        )

        if bandwidth[0, 0] == 0 and bandwidth[1, 1] == 0 and bandwidth[2, 2] == 0:
            # null error case, assign all weight to the closest voxel
            mask = jnp.sqrt(jnp.sum(distances_to_voxel**2, axis=1)) == 0
            normalized_kernel = jnp.where(mask, 1.0, 0.0)
        else:

            kernel = jnp.exp(
                -0.5
                * jnp.sum(
                    (distances_to_voxel @ jnp.linalg.inv(bandwidth))
                    * distances_to_voxel,
                    axis=1,
                )
            )
            norm = 1 / ((2 * jnp.pi) ** (1.5) * jnp.sqrt(jnp.linalg.det(bandwidth)))
            normalized_kernel = kernel * norm

    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    if cutoff_type is not None:
        if cutoff_type == "kernel_unnormalized":
            normalized_kernel[kernel < threshold] = 0.0
        elif cutoff_type == "kernel":
            normalized_kernel[normalized_kernel < threshold] = 0.0
        elif cutoff_type == "distance":
            distances = jnp.sqrt(jnp.sum(distances_to_voxel**2, axis=1))
            normalized_kernel[distances > threshold] = 0.0
        else:
            raise ValueError(f"Unsupported cutoff type: {cutoff_type}")

    return normalized_kernel


def _multivariate_kernel_weights_chunk(
    data_positions,
    bandwidths,
    grid_positions,
    grid_size,
    kernel="gaussian",
    cutoff_type=None,
    threshold=1e-5,
):
    """Evaluate multivariate kernel weights of several objects on a grid.

    Batched version of :func:`multivariate_kernel_density_estimation`, written
    without in-place assignment so that it runs on numpy and JAX.

    Args:
        data_positions (array-like): Object centres, shape ``(N, 3)``.
        bandwidths (array-like): Kernel covariance matrices, shape ``(N, 3, 3)``.
        grid_positions (array-like): Voxel centres, shape ``(M, 3)``.
        grid_size (float): Cell size used to clip distances to voxel boundaries.
        kernel (str): Kernel family to evaluate.
        cutoff_type (str, optional): Optional truncation mode among
            `kernel_unnormalized`, `kernel`, or `distance`.
        threshold (float): Threshold used by `cutoff_type`.

    Returns:
        array: Kernel weights with shape ``(N, M)``.

    Raises:
        ValueError: If the kernel or cutoff type is unsupported.
    """
    if kernel != "gaussian":
        raise ValueError(f"Unsupported kernel: {kernel}")

    distances_to_voxel = jnp.maximum(
        jnp.abs(data_positions[:, None, :] - grid_positions[None, :, :])
        - grid_size / 2,
        0,
    )
    distances = jnp.sqrt(jnp.sum(distances_to_voxel**2, axis=2))

    # Null error case: all the weight goes to the voxel containing the object.
    null_error = jnp.all(jnp.diagonal(bandwidths, axis1=1, axis2=2) == 0, axis=1)
    null_kernel = (distances == 0).astype(distances.dtype)
    bandwidths = jnp.where(null_error[:, None, None], jnp.eye(3), bandwidths)

    unnormalized_kernel = jnp.exp(
        -0.5
        * jnp.sum(
            (distances_to_voxel @ jnp.linalg.inv(bandwidths)) * distances_to_voxel,
            axis=2,
        )
    )
    norm = 1 / ((2 * jnp.pi) ** (1.5) * jnp.sqrt(jnp.linalg.det(bandwidths)))
    unnormalized_kernel = jnp.where(
        null_error[:, None], null_kernel, unnormalized_kernel
    )
    normalized_kernel = jnp.where(
        null_error[:, None], null_kernel, unnormalized_kernel * norm[:, None]
    )

    if cutoff_type is not None:
        if cutoff_type == "kernel_unnormalized":
            cut = unnormalized_kernel < threshold
        elif cutoff_type == "kernel":
            cut = normalized_kernel < threshold
        elif cutoff_type == "distance":
            cut = distances > threshold
        else:
            raise ValueError(f"Unsupported cutoff type: {cutoff_type}")
        normalized_kernel = jnp.where(cut, 0.0, normalized_kernel)

    return normalized_kernel


if jax_installed:
    _multivariate_kernel_weights_chunk = jit(
        _multivariate_kernel_weights_chunk, static_argnames=("kernel", "cutoff_type")
    )


def create_sampled_grid(
    data_position_kernel,
    rcom_max,
    overhead,
    grid_size,
    kind,
    seed,
):
    """Draw one random position per object and paint them onto a mesh.

    For each object its kernel is a block of rows (one row per count). The
    kernels are stacked, and a single ``random.randint`` call draws one row
    offset per object, with the upper bound set to each object's row count.
    The resulting ``(N_objects, 3)`` array of Cartesian positions is then
    painted with unit weights.

    Args:
        data_position_kernel (list[numpy.ndarray]): One array per object.
            Each array has shape ``(number_of_counts, 3)`` with columns
            ``(x, y, z)`` already in Cartesian coordinates.
        rcom_max (float): Maximum comoving radius of the retained region.
        overhead (float): Padding added to the mesh extent.
        grid_size (float): Cell size of the mesh.
        kind (str): Assignment scheme passed to ``create_mesh``.

    Returns:
        numpy.ndarray: Painted mesh values with shape equal to the mesh grid.
    """
    if seed is not None:
        if jax_installed:
            jax_random_key = seed
        else:
            jnp.random.seed(seed)
    else:
        if jax_installed:
            jax_random_key = jax.random.key(int(time.time() * 1e6))

    # Stack all kernels: object i owns rows starts[i] to starts[i] + counts[i].
    counts = jnp.array([len(kernel_rows) for kernel_rows in data_position_kernel])
    starts = jnp.concatenate(
        [jnp.zeros(1, dtype=counts.dtype), jnp.cumsum(counts)[:-1]]
    )
    kernel_stack = jnp.concatenate(
        [jnp.asarray(kernel_rows)[:, :3] for kernel_rows in data_position_kernel]
    )

    if jax_installed:
        jax_random_key, k = jax.random.split(jax_random_key)
        choice_offsets = jax.random.randint(
            k, shape=counts.shape, minval=0, maxval=counts
        )
    else:
        choice_offsets = jnp.random.randint(0, counts)
    sampled_positions = kernel_stack[starts + choice_offsets]

    number_of_objects = counts.shape[0]
    data_weights = jnp.ones(number_of_objects)

    mesh_data_random = create_mesh(
        sampled_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=data_weights,
    )
    return mesh_data_random.value


def grid_data_density_kernel_sampling(
    data_position_sky_kernel,
    rcom_max,
    grid_size,
    grid_type,
    kind,
    Nsampling=100,
    n_subprocess_sampling=16,
    Nrandom=10,
    random_method="cartesian_max_coordinates",
    coord_randoms=None,
    overhead=20,
    d_rcom=10.0,
    seed=None,
):
    """Grid data with pypower and compute density contrast on a mesh.

    Args:
        raobj (array-like): Right ascensions.
        decobj (array-like): Declinations.
        rcomobj (array-like): Comoving distances.
        rcom_max (float): Outer cutoff for grid.
        grid_size (float): Cell size.
        grid_type (str): `rect` or `sphere` cut behavior.
        kind (str): Resampler passed to CatalogMesh.
        Nrandom (int): Randoms per data object.
        random_method (str): Random generation method.
        interlacing (int): Interlacing factor.
        compensate (bool): Apply resampler compensation.
        coord_randoms (tuple, optional): Randoms coordinates for `file` method.
        min_count_random (int): Minimum random count for valid error.
        overhead (float): Extra margin around cutoff.

    Returns:
        dict: Grid with positions, density contrast, errors, and counts.
    """

    kind = kind.lower()
    if kind not in _grid_kind_avail:
        allowed = ", ".join(_grid_kind_avail)
        raise ValueError(f"INVALID GRID TYPE! Allowed kinds: {allowed}")

    data_position_kernel, randoms_positions, _ = prepare_data_position_kernel(
        data_position_sky_kernel,
        rcom_max,
        overhead,
        random_method=random_method,
        Nrandom=Nrandom,
        coord_randoms=coord_randoms,
        seed=seed,
        d_rcom=d_rcom,
    )

    data_weights = jnp.ones(len(data_position_kernel))

    create_sub_grid = partial(
        create_sampled_grid,
        data_position_kernel,
        rcom_max,
        overhead,
        grid_size,
        kind,
    )
    if seed is not None:
        if jax_installed:
            jax_random_key = jax.random.key(seed)
            seeds = jax.random.split(jax_random_key, num=Nsampling)
        else:
            seeds = jnp.random.SeedSequence(seed).generate_state(Nsampling)
    else:
        if jax_installed:
            jax_random_key = jax.random.key(int(time.time() * 1e6))
            seeds = jax.random.split(jax_random_key, num=Nsampling)
        else:
            # Distinct seeds: forked workers would otherwise share the
            # parent's numpy random state and draw identical samples.
            seeds = jnp.random.SeedSequence().generate_state(Nsampling)

    if n_subprocess_sampling > 1:
        if jax_installed:
            vmap_create_sub_grid = jax.vmap(create_sub_grid, in_axes=0)
            mesh_data_random_samples = vmap_create_sub_grid(seeds)
        else:
            with mp.Pool(n_subprocess_sampling) as pool:
                mesh_data_random_samples = pool.map(create_sub_grid, seeds)
    else:
        if jax_installed:
            mesh_data_random_samples = jax.lax.map(create_sub_grid, seeds)
        else:
            mesh_data_random_samples = [
                create_sub_grid(seeds[i]) for i in range(Nsampling)
            ]

    average_mesh_data = jnp.nanmean(mesh_data_random_samples, axis=0)
    std_mesh_data = jnp.nanstd(mesh_data_random_samples, axis=0)

    randoms_weights = jnp.ones((randoms_positions.shape[0],))
    scaling = jnp.sum(data_weights) / jnp.sum(randoms_weights)
    mesh_randoms = create_mesh(
        randoms_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=randoms_weights,
        scaling=scaling,
    )

    mesh_count_randoms = create_mesh(
        randoms_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement="ngp",
        weights=randoms_weights,
    )
    count_randoms = jnp.ravel(mesh_count_randoms.value)

    mask_randoms_nonzero = mesh_randoms.value != 0.0
    safe_randoms = jnp.where(mask_randoms_nonzero, mesh_randoms.value, 1.0)
    density_contrast = jnp.ravel(
        jnp.where(mask_randoms_nonzero, average_mesh_data / safe_randoms - 1, jnp.nan)
    )
    density_contrast_err = jnp.ravel(
        jnp.where(mask_randoms_nonzero, std_mesh_data / safe_randoms, jnp.nan)
    )

    xgrid, ygrid, zgrid, ragrid, decgrid, rcomgrid = define_grid_from_mesh(
        mesh_randoms,
        grid_size,
    )

    grid = {
        "x": xgrid,
        "y": ygrid,
        "z": zgrid,
        "ra": ragrid,
        "dec": decgrid,
        "rcom_zobs": rcomgrid,
        "density": density_contrast,
        "density_error": density_contrast_err,
        "count_random": count_randoms,
        "count_random_scaled": count_randoms * scaling,
    }

    cut_grid_type(
        grid,
        grid_type,
        rcom_max,
    )

    return grid


def grid_data_density_kernel_sum(
    data_position_sky_kernel,
    rcom_max,
    grid_size,
    grid_type,
    kind,
    Nrandom=10,
    random_method="cartesian_max_coordinates",
    coord_randoms=None,
    min_count_random=0,
    overhead=20,
    seed=None,
    data_weights=None,
    randoms_weights=None,
    cut_grid=False,
    d_rcom=10.0,
    n_bins_sin_dec=1,
):
    """Grid data kernels (e.g. GW localisations) by summing their samples.

    Each kernel carries a total weight of 1 (times `data_weights`), spread
    over its samples. The density contrast is data / randoms - 1 and its error
    is Poisson on the expected number of objects per cell.

    Args:
        data_position_sky_kernel (list or array-like): Kernels of shape
            `(3, n_i)` with rows `(ra, dec, rcom)`; sizes may differ. Samples
            must lie in the cube `|x|, |y|, |z| < rcom_max` (painting is
            periodic).
        rcom_max (float): Half side of the cube.
        grid_size (float): Cell size.
        grid_type (str): `rect` or `sphere` cut behavior.
        kind (str): Mass assignment of the data and randoms.
        Nrandom (int): Randoms per kernel; total number for
            `cartesian_max_coordinates`.
        random_method (str): A `define_randoms` method, `choice_kernel` for
            random kernels, or `analytic` for expected counts without randoms
            (`compute_random_counts_analytic`).
        coord_randoms (array-like, optional): Randoms for the `file` method.
        min_count_random (float): Minimum random count (expected count for
            `analytic`) for a valid error.
        overhead (float): Extra margin around the cube.
        seed (int, optional): Random seed.
        data_weights (array-like, optional): One weight per kernel.
        randoms_weights (array-like, optional): One weight per random.
        cut_grid (bool): Apply the `grid_type` cut.
        d_rcom (float): rcom window of the kernel choice (`choice_kernel`) and
            rcom bin width (`analytic`).
        n_bins_sin_dec (int): Number of sin(dec) bins (`analytic`).

    Returns:
        dict: Grid with positions, density contrast, errors, and counts.
    """

    kind = kind.lower()
    if kind not in _grid_kind_avail:
        allowed = ", ".join(_grid_kind_avail)
        raise ValueError(f"INVALID GRID TYPE! Allowed kinds: {allowed}")

    data_position_kernel, randoms_positions, kernel_randoms = (
        prepare_data_position_kernel(
            data_position_sky_kernel,
            rcom_max,
            overhead,
            random_method=None if random_method == "analytic" else random_method,
            Nrandom=Nrandom,
            coord_randoms=coord_randoms,
            seed=seed,
            d_rcom=d_rcom,
        )
    )

    counts = jnp.array([len(kernel_rows) for kernel_rows in data_position_kernel])
    if data_weights is None:
        data_weights_sum = jnp.repeat(1.0 / counts, counts)
    else:
        data_weights_sum = jnp.repeat(data_weights / counts, counts)

    # np.vstack: jnp.vstack on a list of numpy arrays is quadratic in the list length
    data_position_kernel_sum = jnp.asarray(np.vstack(data_position_kernel))

    mesh_data = create_mesh(
        data_position_kernel_sum,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=data_weights_sum,
    )

    xgrid, ygrid, zgrid, ragrid, decgrid, rcomgrid = define_grid_from_mesh(
        mesh_data,
        grid_size,
    )

    grid = {
        "x": xgrid,
        "y": ygrid,
        "z": zgrid,
        "ra": ragrid,
        "dec": decgrid,
        "rcom_zobs": rcomgrid,
        "value_data": jnp.ravel(mesh_data.value),
    }

    if random_method == "analytic":
        ra_kernel, dec_kernel, rcom_kernel = np.concatenate(
            [np.asarray(kernel) for kernel in data_position_sky_kernel], axis=1
        )
        count_randoms_value = compute_random_counts_analytic(
            ra_kernel,
            dec_kernel,
            rcom_kernel,
            data_weights_sum,
            xgrid,
            ygrid,
            zgrid,
            grid_size,
            rcom_max,
            d_rcom=d_rcom,
            n_bins_sin_dec=n_bins_sin_dec,
        )
        count_randoms_value = jnp.asarray(
            count_randoms_value.reshape(mesh_data.value.shape)
        )
        randoms_value = count_randoms_value
        scaling = 1.0

    elif kernel_randoms is not None:
        # Random kernels come in chunks: paint each chunk and accumulate, so
        # that all random samples never sit in memory at once. Chunks are
        # zero-padded to a multiple of 2**20 samples so that painting compiles
        # for a few shapes only.
        randoms_value, count_randoms_value, sum_random_weights = 0.0, 0.0, 0.0
        for positions_chunk, sizes_chunk, start in kernel_randoms:
            if randoms_weights is None:
                weights_kernel = 1.0 / sizes_chunk
            else:
                weights_kernel = (
                    np.asarray(randoms_weights)[start : start + len(sizes_chunk)]
                    / sizes_chunk
                )
            weights_chunk = np.repeat(weights_kernel, sizes_chunk)
            sum_random_weights += weights_chunk.sum()
            n_pad = -len(weights_chunk) % 2**20
            positions_chunk = np.pad(positions_chunk, ((0, n_pad), (0, 0)))
            weights_chunk = np.pad(weights_chunk, (0, n_pad))
            randoms_value += create_mesh(
                positions_chunk,
                2 * (rcom_max + overhead),
                grid_size,
                assignement=kind,
                weights=weights_chunk,
            ).value
            count_randoms_value += create_mesh(
                positions_chunk,
                2 * (rcom_max + overhead),
                grid_size,
                assignement="ngp",
                weights=weights_chunk,
            ).value

        scaling = jnp.sum(data_weights_sum) / sum_random_weights
        randoms_value = randoms_value * scaling

    else:
        if randoms_weights is None:
            randoms_weights = jnp.ones((randoms_positions.shape[0],))

        scaling = jnp.sum(data_weights_sum) / jnp.sum(randoms_weights)
        randoms_value = create_mesh(
            randoms_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement=kind,
            weights=randoms_weights,
            scaling=scaling,
        ).value
        count_randoms_value = create_mesh(
            randoms_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement="ngp",
            weights=randoms_weights,
        ).value

    mask_randoms_nonzero = randoms_value != 0.0
    safe_randoms = jnp.where(mask_randoms_nonzero, randoms_value, 1.0)
    density_contrast = jnp.ravel(
        jnp.where(mask_randoms_nonzero, mesh_data.value / safe_randoms - 1, jnp.nan)
    )
    count_randoms = jnp.ravel(count_randoms_value)
    mask = count_randoms > min_count_random
    safe_count_randoms = jnp.where(mask, count_randoms, 1)
    density_contrast_err = jnp.where(
        mask, jnp.sqrt(1 / (safe_count_randoms * scaling)), jnp.nan
    )

    grid["density"] = density_contrast
    grid["density_error"] = density_contrast_err
    grid["count_random"] = count_randoms
    grid["count_random_scaled"] = count_randoms * scaling

    if cut_grid:
        cut_grid_type(
            grid,
            grid_type,
            rcom_max,
        )

    return grid


def grid_data_velocity(
    data_position_sky,
    rcom_max,
    grid_size,
    grid_type,
    kind,
    variance,
    velocity=None,
    overhead=20,
    count_weights=None,
):
    """Grid velocity data and compute per-cell means and errors.

    Args:
        data_position_sky (array-like): Sky coordinates with columns `(ra, dec, rcom)`.
        rcom_max (float): Outer cutoff.
        grid_size (float): Cell size.
        grid_type (str): `rect` or `sphere` cut behavior.
        kind (str): Assignment kernel among the supported mesh schemes.
        variance (array-like): Per-object variances.
        velocity (array-like|None): Per-object velocities when computing the
            weighted mean velocity.
        overhead (float): Extra margin around cutoff.
        count_weights (array-like): Weights in the count of the tracers.

    Returns:
        dict: Grid with positions, velocity (optional), variance, and counts.
    """

    data_positions, _, _, mask_data, _ = prepare_data_position(
        data_position_sky,
        rcom_max,
        overhead,
    )
    if count_weights is None:
        count_weights = jnp.ones((data_positions.shape[0],))
    else:
        count_weights = count_weights[mask_data]
    variance = variance[mask_data]

    mesh_variance = create_mesh(
        data_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=variance,
    )

    mesh_count = create_mesh(
        data_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=count_weights,
    )

    N_in_cell = jnp.ravel(mesh_count.value)
    variance_grid = jnp.ravel(mesh_variance.value) / (N_in_cell**2)

    if velocity is not None:
        velocity = velocity[mask_data]

        weights_weighted_mean = velocity / variance

        mesh_weighted_velocity = create_mesh(
            data_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement=kind,
            weights=weights_weighted_mean,
        )

        mesh_inverse_variance = create_mesh(
            data_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement=kind,
            weights=1 / variance,
        )
        velocity_grid = jnp.ravel(mesh_weighted_velocity.value) / jnp.ravel(
            mesh_inverse_variance.value
        )
    else:
        velocity_grid = None

    xgrid, ygrid, zgrid, ragrid, decgrid, rcomgrid = define_grid_from_mesh(
        mesh_variance,
        grid_size,
    )
    grid = {
        "x": xgrid,
        "y": ygrid,
        "z": zgrid,
        "ra": ragrid,
        "dec": decgrid,
        "rcom_zobs": rcomgrid,
        "velocity_error": jnp.sqrt(variance_grid),
        "N_in_cell": N_in_cell,
    }
    if velocity is not None:
        grid["velocity"] = velocity_grid

    if grid_type == "rect":
        cut_grid(
            grid,
            remove_nan_density=True,
            remove_empty_cells=True,
            xmax=rcom_max,
            ymax=rcom_max,
            zmax=rcom_max,
            remove_origin=True,
        )

    if grid_type == "sphere":
        cut_grid(
            grid,
            remove_nan_density=True,
            remove_empty_cells=True,
            rcom_max=rcom_max,
            remove_origin=True,
        )

    return grid
