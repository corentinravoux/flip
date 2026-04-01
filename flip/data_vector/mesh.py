try:
    from pmesh.pm import ParticleMesh
except ImportError:
    ParticleMesh = None
import multiprocessing as mp

import numpy as np

from flip import utils

_grid_kind_avail = ["ngp", "ngp_errw", "cic", "tsc", "pcs"]

_grid_order_dict = {
    "ngp": 1,
    "ngp_errw": 1,
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
    window = np.zeros_like(k)
    theta = np.linspace(0, np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    kx = np.outer(np.sin(theta), np.cos(phi))
    ky = np.outer(np.sin(theta), np.sin(phi))
    kz = np.outer(np.cos(theta), np.ones(n))

    # Forgotten in Howlett et al. formula
    # we add spherical coordinate solid angle element
    dthetaphi = np.outer(np.sin(theta), np.ones(phi.size))
    for i in range(k.size):
        # the factor here has an extra np.pi because of the definition of np.sinc
        fact = (k[i] * grid_size) / (2 * np.pi)
        func = (
            np.sinc(fact * kx) * np.sinc(fact * ky) * np.sinc(fact * kz)
        ) ** order * dthetaphi
        win_theta = np.trapz(func, x=phi)
        window[i] = np.trapz(win_theta, x=theta)
    window *= 1 / (4 * np.pi)
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
    abs_ds = np.abs(ds)
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
    abs_ds = np.abs(ds)
    w = (1.0 - abs_ds) * (abs_ds < 1)
    return w


def tsc_weight(ds):
    """Return triangular-shaped-cloud assignment weights.

    Args:
        ds (array-like): Distance to the cell center in cell-size units.

    Returns:
        numpy.ndarray: TSC weights evaluated at `ds`.
    """
    abs_ds = np.abs(ds)
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
    abs_ds = np.abs(ds)
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


def define_randoms(
    random_method,
    xobj=None,
    yobj=None,
    zobj=None,
    raobj=None,
    decobj=None,
    rcomobj=None,
    Nrandom=None,
    coord_randoms=None,
    max_coordinates=None,
    seed=None,
):
    """Generate random Cartesian positions for density gridding.

    Supports uniform Cartesian sampling, random catalogs drawn from the
    observed angular and radial distributions, or externally supplied randoms.

    Args:
        random_method (str): `cartesian`, `choice`, `choice_redshift`, or `file`.
        xobj (array-like): Data x positions.
        yobj (array-like): Data y positions.
        zobj (array-like): Data z positions.
        raobj (array-like): Data right ascensions.
        decobj (array-like): Data declinations.
        rcomobj (array-like): Data comoving distances.
        Nrandom (int): Number of randoms per object.
        coord_randoms (tuple, optional): `(ra, dec, rcom)` for `file` method.
        max_coordinates (float, optional): Coordinate cutoff for `file` method.
        seed (int, optional): Random seed used for reproducible draws.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Random x, y, z positions.
    """

    def _needed_params(needed_parameter):
        for p in needed_parameter:
            if p is None:
                raise ValueError(
                    f"random_method={random_method} requires {p} to be provided"
                )

    if seed is not None:
        np.random.seed(seed)

    # Uniform in X,Y,Z
    if random_method == "cartesian":
        _needed_params([xobj, yobj, zobj])
        N = xobj.size
        xobj_random = (np.max(xobj) - np.min(xobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(xobj)
        yobj_random = (np.max(yobj) - np.min(yobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(yobj)
        zobj_random = (np.max(zobj) - np.min(zobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(zobj)

    elif random_method == "cartesian_max_coordinates":
        xobj_random = (2 * max_coordinates) * np.random.random(
            size=Nrandom
        ) - max_coordinates
        yobj_random = (2 * max_coordinates) * np.random.random(
            size=Nrandom
        ) - max_coordinates
        zobj_random = (2 * max_coordinates) * np.random.random(
            size=Nrandom
        ) - max_coordinates

    # Choice in the ra, dec, and redshift data coordinates to create the random.
    elif random_method == "choice":
        _needed_params([raobj, decobj, rcomobj])
        N = raobj.size
        counts_ra, bins_ra = np.histogram(raobj, bins=1000)
        ra_random = np.random.choice(
            (bins_ra[1:] + bins_ra[:-1]) / 2,
            p=counts_ra / float(counts_ra.sum()),
            size=Nrandom * N,
        )
        counts_dec, bins_dec = np.histogram(decobj, bins=1000)
        dec_random = np.random.choice(
            (bins_dec[1:] + bins_dec[:-1]) / 2,
            p=counts_dec / float(counts_dec.sum()),
            size=Nrandom * N,
        )
        counts_rcom, bins_rcom = np.histogram(rcomobj, bins=1000)
        rcom_random = np.random.choice(
            (bins_rcom[1:] + bins_rcom[:-1]) / 2,
            p=counts_rcom / float(counts_rcom.sum()),
            size=Nrandom * N,
        )

        xobj_random, yobj_random, zobj_random = utils.radec2cart(
            rcom_random, ra_random, dec_random
        )

    # Uniform in ra and dec. Choice in the redshift data coordinates.
    elif random_method == "choice_redshift":
        _needed_params([raobj, decobj, rcomobj])
        N = raobj.size
        ra_random = (np.max(raobj) - np.min(raobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(raobj)
        dec_random = (np.max(decobj) - np.min(decobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(decobj)
        counts_rcom, bins_rcom = np.histogram(rcomobj, bins=1000)
        rcom_random = np.random.choice(
            (bins_rcom[1:] + bins_rcom[:-1]) / 2,
            p=counts_rcom / float(counts_rcom.sum()),
            size=Nrandom * N,
        )

        xobj_random, yobj_random, zobj_random = utils.radec2cart(
            rcom_random, ra_random, dec_random
        )

    # From random file
    elif random_method == "file":
        _needed_params([coord_randoms])
        ra_random, dec_random, rcom_random = (
            coord_randoms[0],
            coord_randoms[1],
            coord_randoms[2],
        )

        xobj_random, yobj_random, zobj_random = utils.radec2cart(
            rcom_random, ra_random, dec_random
        )
        if max_coordinates is not None:
            mask_random = np.abs(xobj_random) < max_coordinates
            mask_random &= np.abs(yobj_random) < max_coordinates
            mask_random &= np.abs(zobj_random) < max_coordinates
            xobj_random = xobj_random[mask_random]
            yobj_random = yobj_random[mask_random]
            zobj_random = zobj_random[mask_random]

    return xobj_random, yobj_random, zobj_random


def create_mesh(
    positions,
    boxsize,
    cellsize,
    assignement="ngp",
    weights=None,
    scaling=None,
):
    """Paint weighted positions onto a `pmesh` mesh.

    Args:
        positions (array-like): Cartesian positions with shape `(N, 3)`.
        boxsize (float): Box size used to create the mesh.
        cellsize (float): Target cell size.
        assignement (str): Assignment scheme among `ngp`, `ngp_errw`, `cic`,
            `tsc`, and `pcs`.
        weights (array-like, optional): Per-object weights to paint on the mesh.
        scaling (float, optional): Extra multiplicative factor applied to weights.

    Returns:
        pmesh.pm.RealField: Painted real-space mesh.

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

    pm = ParticleMesh(
        BoxSize=boxsize,
        Nmesh=nmesh,
        dtype="f8",
    )
    offset = boxcenter - boxsize / 2.0
    _slab_npoints_max = int(1024 * 1024 * 4)

    def paint(positions, weights, scaling, out, transform=None):
        positions = positions - offset
        factor = 0.5
        if not np.isfinite(positions).all():
            raise ValueError("Some positions are NaN/inf")
        if not np.isfinite(weights).all():
            raise ValueError("Some weights are NaN/inf")
        if scaling is not None:
            weights = weights * scaling

        def paint_slab(sl):
            p = positions[sl]
            size = len(p)
            layout = pm.decompose(p, smoothing=factor * _grid_order_dict[assignement])
            p = layout.exchange(p)
            w = layout.exchange(weights[sl])
            pm.paint(
                p,
                mass=w,
                resampler=resampler,
                transform=transform,
                hold=True,
                out=out,
            )
            return size

        islab = 0
        slab_npoints = _slab_npoints_max
        while islab < len(positions):
            sl = slice(islab, islab + slab_npoints)
            _ = paint_slab(sl)
            islab += slab_npoints
            slab_npoints = min(_slab_npoints_max, int(slab_npoints * 1.2))

    out = pm.create(type="real", value=0.0)
    paint(positions, weights, scaling, out)

    return out


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
    mask = np.full(grid["ra"].shape, True)
    if n_cut is not None:
        mask &= grid["nincell"] > n_cut
    if weight_min is not None:
        mask &= grid["sum_weights"] > weight_min
    if rcom_max is not None:
        mask &= grid["rcom_zobs"] < rcom_max
    if xmax is not None:
        mask &= np.abs(grid["x"]) < xmax
    if ymax is not None:
        mask &= np.abs(grid["y"]) < ymax
    if zmax is not None:
        mask &= np.abs(grid["z"]) < zmax
    if remove_nan_density:
        if "density" in grid:
            mask &= ~(np.isnan(grid["density"]))
        if "density_error" in grid:
            mask &= ~(np.isnan(grid["density_error"]))
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
):

    data_position_kernel = []
    for i in range(len(data_position_sky_kernel)):
        kernel = data_position_sky_kernel[i]
        x_kernel, y_kernel, z_kernel = utils.radec2cart(
            kernel[:, 0],
            kernel[:, 1],
            kernel[:, 2],
        )
        kernel_cartesian = np.array([x_kernel, y_kernel, z_kernel, kernel[:, 3]]).T
        data_position_kernel.append(kernel_cartesian)

    if random_method is not None:
        (
            xobj_random,
            yobj_random,
            zobj_random,
        ) = define_randoms(
            random_method,
            Nrandom=Nrandom,
            coord_randoms=coord_randoms,
            max_coordinates=rcom_max + overhead,
            seed=seed,
        )
        randoms_positions = np.array([xobj_random, yobj_random, zobj_random]).T
    else:
        randoms_positions = None

    return data_position_kernel, randoms_positions


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
    xobj, yobj, zobj = utils.radec2cart(rcomobj, raobj, decobj)
    mask = np.abs(xobj) < rcom_max + overhead
    mask &= np.abs(yobj) < rcom_max + overhead
    mask &= np.abs(zobj) < rcom_max + overhead
    xobj, yobj, zobj = xobj[mask], yobj[mask], zobj[mask]

    if data_position_sky_bandwidth is not None:

        jacobian = utils.radec2cart_jacobian(rcomobj[mask], raobj[mask], decobj[mask])

        data_position_sky_bandwidth = data_position_sky_bandwidth[mask, :, :]

        data_position_bandwith = (
            jacobian.T
            @ data_position_sky_bandwidth
            @ np.transpose(jacobian.T, (0, 2, 1))
        )
    else:
        data_position_bandwith = None

    raobj, decobj, rcomobj = raobj[mask], decobj[mask], rcomobj[mask]
    if random_method is not None:
        (
            xobj_random,
            yobj_random,
            zobj_random,
        ) = define_randoms(
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
        randoms_positions = np.array([xobj_random, yobj_random, zobj_random]).T
    else:
        randoms_positions = None

    data_positions = np.array([xobj, yobj, zobj]).T

    return (
        data_positions,
        data_position_bandwith,
        randoms_positions,
    )


def define_grid_from_mesh(mesh_data, grid_size):
    """Extract voxel-center coordinates and sky coordinates from a mesh.

    Args:
        mesh_data: `pmesh` field used to define the grid geometry.
        grid_size (float): Cell size used to shift coordinates to voxel centers.

    Returns:
        tuple[numpy.ndarray, ...]: Cartesian and sky coordinates of all grid cells.
    """
    coord_mesh = np.array(
        np.meshgrid(
            np.sort(mesh_data.x[0][:, 0, 0]),
            np.sort(mesh_data.x[1][0, :, 0]),
            np.sort(mesh_data.x[2][0, 0, :]),
            indexing="ij",
        )
    )
    xgrid = np.ravel(coord_mesh[0, :, :, :]) + grid_size / 2
    ygrid = np.ravel(coord_mesh[1, :, :, :]) + grid_size / 2
    zgrid = np.ravel(coord_mesh[2, :, :, :]) + grid_size / 2

    rcomgrid, ragrid, decgrid = utils.cart2radec(xgrid, ygrid, zgrid)

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

    data_positions, _, randoms_positions = prepare_data_position(
        data_position_sky,
        rcom_max,
        overhead,
        random_method=random_method,
        Nrandom=Nrandom,
        coord_randoms=coord_randoms,
        seed=seed,
    )

    data_weights = np.ones((data_positions.shape[0],))
    randoms_weights = np.ones((randoms_positions.shape[0],))

    mesh_data = create_mesh(
        data_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=data_weights,
    )

    mesh_randoms = create_mesh(
        randoms_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=randoms_weights,
        scaling=np.sum(data_weights) / np.sum(randoms_weights),
    )
    if kind == "ngp":
        mesh_count_randoms = mesh_randoms.copy()
    else:
        mesh_count_randoms = create_mesh(
            randoms_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement="ngp",
            weights=randoms_weights,
            scaling=np.sum(data_weights) / np.sum(randoms_weights),
        )

    density_contrast = np.zeros_like(mesh_data.value)
    mask_randoms_nonzero = mesh_randoms.value != 0.0
    density_contrast[mask_randoms_nonzero] = np.ravel(
        mesh_data.value[mask_randoms_nonzero] / mesh_randoms.value[mask_randoms_nonzero]
        - 1
    )
    density_contrast[~mask_randoms_nonzero] = np.nan
    density_contrast = np.ravel(density_contrast)

    count_randoms = np.ravel(mesh_count_randoms.value).astype(int)
    density_contrast_err = np.full(count_randoms.shape, np.nan)
    mask = count_randoms > min_count_random
    density_contrast_err[mask] = np.sqrt(1 / (count_randoms[mask]))

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
        "density": density_contrast,
        "density_error": density_contrast_err,
        "count_random": count_randoms,
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

    data_positions, data_position_bandwith, randoms_positions = prepare_data_position(
        data_position_sky,
        rcom_max,
        overhead,
        random_method=random_method,
        Nrandom=Nrandom,
        coord_randoms=coord_randoms,
        seed=seed,
        data_position_sky_bandwidth=data_position_sky_bandwidth,
    )

    data_weights = np.ones((data_positions.shape[0],))
    randoms_weights = np.ones((randoms_positions.shape[0],))

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

    grid_positions = np.array([xgrid, ygrid, zgrid]).T

    data_kernel_positions = []
    data_kernel_weights = []
    for i in range(data_positions.shape[0]):
        kernel_weights = multivariate_kernel_density_estimation(
            data_positions[i],
            data_position_bandwith[i],
            grid_positions,
            grid_size,
            kernel=kernel,
            cutoff_type=cutoff_type,
            threshold=threshold,
        )
        mask = kernel_weights != 0.0
        data_kernel_positions.append(grid_positions[mask])
        data_kernel_weights.append(kernel_weights[mask] / np.sum(kernel_weights[mask]))

    data_kernel_positions = np.concatenate(data_kernel_positions, axis=0)
    data_kernel_weights = np.concatenate(data_kernel_weights, axis=0)

    data_kernel_weights = (
        data_positions.shape[0] * data_kernel_weights / np.sum(data_kernel_weights)
    )

    mesh_data_kernel = create_mesh(
        data_kernel_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=data_kernel_weights,
    )

    mesh_randoms = create_mesh(
        randoms_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=randoms_weights,
        scaling=np.sum(data_weights) / np.sum(randoms_weights),
    )
    if kind == "ngp":
        mesh_count_randoms = mesh_randoms.copy()
    else:
        mesh_count_randoms = create_mesh(
            randoms_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement="ngp",
            weights=randoms_weights,
            scaling=np.sum(data_weights) / np.sum(randoms_weights),
        )

    density_contrast = np.zeros_like(mesh_data_kernel.value)
    mask_randoms_nonzero = mesh_randoms.value != 0.0
    density_contrast[mask_randoms_nonzero] = np.ravel(
        mesh_data_kernel.value[mask_randoms_nonzero]
        / mesh_randoms.value[mask_randoms_nonzero]
        - 1
    )
    density_contrast[~mask_randoms_nonzero] = np.nan
    density_contrast = np.ravel(density_contrast)

    count_randoms = np.ravel(mesh_count_randoms.value).astype(int)
    density_contrast_err = np.full(count_randoms.shape, np.nan)
    mask = count_randoms > min_count_random
    density_contrast_err[mask] = np.sqrt(1 / (count_randoms[mask]))

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

        distances_to_voxel = np.maximum(
            np.abs(data_position - grid_positions) - grid_size / 2, 0
        )

        if bandwidth[0, 0] == 0 and bandwidth[1, 1] == 0 and bandwidth[2, 2] == 0:
            # null error case, assign all weight to the closest voxel
            normalized_kernel = np.zeros((grid_positions.shape[0],))
            mask = np.sqrt(np.sum(distances_to_voxel**2, axis=1)) == 0
            normalized_kernel[mask] = 1.0
        else:

            kernel = np.exp(
                -0.5
                * np.sum(
                    (distances_to_voxel @ np.linalg.inv(bandwidth))
                    * distances_to_voxel,
                    axis=1,
                )
            )
            norm = 1 / ((2 * np.pi) ** (1.5) * np.sqrt(np.linalg.det(bandwidth)))
            normalized_kernel = kernel * norm

    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    if cutoff_type is not None:
        if cutoff_type == "kernel_unnormalized":
            normalized_kernel[kernel < threshold] = 0.0
        elif cutoff_type == "kernel":
            normalized_kernel[normalized_kernel < threshold] = 0.0
        elif cutoff_type == "distance":
            distances = np.sqrt(np.sum(distances_to_voxel**2, axis=1))
            normalized_kernel[distances > threshold] = 0.0
        else:
            raise ValueError(f"Unsupported cutoff type: {cutoff_type}")

    return normalized_kernel


def grid_data_density_kernel_sampling(
    data_position_sky_kernel,
    rcom_max,
    grid_size,
    grid_type,
    kind,
    Nsampling=100,
    n_subprocess_sampling=16,
    Nrandom=10,
    random_method="cartesian",
    coord_randoms=None,
    overhead=20,
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

    (data_position_kernel, randoms_positions) = prepare_data_position_kernel(
        data_position_sky_kernel,
        rcom_max,
        overhead,
        random_method=random_method,
        Nrandom=Nrandom,
        coord_randoms=coord_randoms,
        seed=seed,
    )
    data_weights = np.ones(len(data_position_kernel))

    def create_sub_grid():
        np.random.seed()  # ensure different random states across subprocesses
        data_positions_random = data_position_kernel[
            np.random.choice(
                data_position_kernel.shape[0],
                replace=False,
            )
        ]

        mesh_data_random = create_mesh(
            data_positions_random,
            2 * (rcom_max + overhead),
            grid_size,
            assignement=kind,
            weights=data_weights,
        )
        return mesh_data_random.value

    if n_subprocess_sampling > 1:
        with mp.Pool(n_subprocess_sampling) as pool:
            mesh_data_random_samples = pool.map(create_sub_grid, range(Nsampling))
    else:
        mesh_data_random_samples = [create_sub_grid() for _ in range(Nsampling)]

    average_mesh_data = np.nanmean(mesh_data_random_samples, axis=0)
    std_mesh_data = np.nanstd(mesh_data_random_samples, axis=0)

    randoms_weights = np.ones((randoms_positions.shape[0],))
    mesh_randoms = create_mesh(
        randoms_positions,
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=randoms_weights,
        scaling=np.sum(data_weights) / np.sum(randoms_weights),
    )
    if kind == "ngp":
        mesh_count_randoms = mesh_randoms.copy()
    else:
        mesh_count_randoms = create_mesh(
            randoms_positions,
            2 * (rcom_max + overhead),
            grid_size,
            assignement="ngp",
            weights=randoms_weights,
            scaling=np.sum(data_weights) / np.sum(randoms_weights),
        )
    count_randoms = np.ravel(mesh_count_randoms.value).astype(int)

    density_contrast = np.zeros_like(average_mesh_data.value)
    density_contrast_err = np.zeros_like(average_mesh_data.value)

    mask_randoms_nonzero = mesh_randoms.value != 0.0
    density_contrast[mask_randoms_nonzero] = np.ravel(
        average_mesh_data[mask_randoms_nonzero]
        / mesh_randoms.value[mask_randoms_nonzero]
        - 1
    )
    density_contrast[~mask_randoms_nonzero] = np.nan
    density_contrast_err[mask_randoms_nonzero] = np.ravel(
        std_mesh_data[mask_randoms_nonzero] / mesh_randoms.value[mask_randoms_nonzero]
    )
    density_contrast_err[~mask_randoms_nonzero] = np.nan

    density_contrast = np.ravel(density_contrast)
    density_contrast_err = np.ravel(density_contrast_err)

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
    }

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

    Returns:
        dict: Grid with positions, velocity (optional), variance, and counts.
    """

    data_positions, _, _ = prepare_data_position(
        data_position_sky,
        rcom_max,
        overhead,
    )

    count_weights = np.ones((data_positions.shape[0],))

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

    N_in_cell = np.ravel(mesh_count.value)
    variance_grid = np.ravel(mesh_variance.value) / (N_in_cell**2)

    if velocity is not None:
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
        velocity_grid = np.ravel(mesh_weighted_velocity.value) / np.ravel(
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
        "velocity_error": np.sqrt(variance_grid),
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
