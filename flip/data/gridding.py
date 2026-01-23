import numpy as np

from flip import utils

log = utils.create_log()
try:
    from pypower import CatalogMesh
except ImportError:
    log.add("No pypower module detected, gridding with this method is unavailable")

# CR - No cut in healpix implemented with randoms

_GRID_KIND = ["ngp", "ngp_errw", "cic", "tsc", "pcs"]


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
    _order_dic = {
        "ngp": 1,
        "ngp_errw": 1,
        "cic": 2,
        "tsc": 3,
        "pcs": 4,
    }

    if kind not in _GRID_KIND:
        allowed = ", ".join(_GRID_KIND)
        raise ValueError(f"INVALID GRID TYPE! Allowed kinds: {allowed}")
    if grid_size == 0:
        return None
    return _compute_grid_window(grid_size, kh, _order_dic[kind], n)


def construct_grid_regular_sphere(grid_size, rcom_max):
    """Construct a regular spherical grid of voxel centers.

    Args:
        grid_size (float): Cell size.
        rcom_max (float): Maximum comoving radius cutoff.

    Returns:
        dict: Grid dictionary with keys `ra`, `dec`, `rcom`, `x`, `y`, `z` after cut.
    """

    # Number of grid voxels per axis
    n_grid = np.floor(rcom_max / grid_size).astype(np.int64)

    # Determine the coordinates of the voxel centers
    cp_x, cp_y, cp_z = np.reshape(
        np.meshgrid(
            *(np.linspace(-n_grid * grid_size, n_grid * grid_size, 2 * n_grid + 1),)
            * 3,
            indexing="ij",
        ),
        (3, (2 * n_grid + 1) ** 3),
    )

    # Convert to ra, dec, r_comov
    center_r_comov, center_ra, center_dec = utils.cart2radec(cp_x, cp_y, cp_z)

    # Cut the grid with rcom_max
    grid = {
        "ra": center_ra,
        "dec": center_dec,
        "rcom_zobs": center_r_comov,
        "x": cp_x,
        "y": cp_y,
        "z": cp_z,
    }
    cut_grid(grid, n_cut=None, weight_min=None, rcom_max=rcom_max)

    return grid


def construct_grid_regular_rectangular(grid_size, rcom_max):
    """Construct a regular rectangular grid of voxel centers.

    Args:
        grid_size (float): Cell size.
        rcom_max (float): Half-side length of the cube.

    Returns:
        dict: Grid dictionary with `ra`, `dec`, `rcom`, `x`, `y`, `z`.
    """

    # Number of grid voxels per axis
    n_grid = np.floor(rcom_max / grid_size).astype(np.int64)

    # Determine the coordinates of the voxel centers
    cp_x, cp_y, cp_z = np.reshape(
        np.meshgrid(
            *(np.linspace(-n_grid * grid_size, n_grid * grid_size, 2 * n_grid + 1),)
            * 3,
            indexing="ij",
        ),
        (3, (2 * n_grid + 1) ** 3),
    )

    # Convert to ra, dec, r_comov
    center_r_comov, center_ra, center_dec = utils.cart2radec(cp_x, cp_y, cp_z)

    # Cut the grid with rcom_max
    grid = {
        "ra": center_ra,
        "dec": center_dec,
        "rcom_zobs": center_r_comov,
        "x": cp_x,
        "y": cp_y,
        "z": cp_z,
    }
    return grid


def ngp_weight(ds):
    """Nearest Grid Point."""
    abs_ds = np.abs(ds)
    w = 1.0 * (abs_ds < 1 / 2)
    return w


def ngp_errw_weight(ds):
    """Nearest Grid Point with Weighted error."""
    return ngp_weight(ds)


def cic_weight(ds):
    """Cloud In Cell."""
    abs_ds = np.abs(ds)
    w = (1.0 - abs_ds) * (abs_ds < 1)
    return w


def tsc_weight(ds):
    """Triangular Shaped Cloud."""
    abs_ds = np.abs(ds)
    w = (3 / 4 - ds**2) * (abs_ds < 1 / 2)
    w += 1 / 2 * (3 / 2 - abs_ds) ** 2 * ((1 / 2 <= abs_ds) & (abs_ds < 3 / 2))
    return w


def pcs_weight(ds):
    """Triangular Shaped Cloud."""
    abs_ds = np.abs(ds)
    w = 1 / 6 * (4 - 6 * ds**2 + 3 * abs_ds**3) * (abs_ds < 1)
    w += 1 / 6 * (2 - abs_ds) ** 3 * ((1 <= abs_ds) & (abs_ds < 2))
    return w


def attribute_weight_density(
    grid_size,
    xobj,
    yobj,
    zobj,
    xgrid,
    ygrid,
    zgrid,
    weight_fun,
):
    """Accumulate assignment weights and counts for objects onto a grid.

    Args:
        grid_size (float): Cell size.
        xobj (array-like): Object x positions.
        yobj (array-like): Object y positions.
        zobj (array-like): Object z positions.
        xgrid (array-like): Grid x centers.
        ygrid (array-like): Grid y centers.
        zgrid (array-like): Grid z centers.
        weight_fun (Callable): Resampler weight function of distance.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
            sum of weights, sum of squared weights, and objects-per-cell counts.
    """

    Nobj = len(xobj)
    Ngrid = len(xgrid)

    # Init array
    grid_val = np.zeros(Ngrid, dtype="float")
    grid_var = np.zeros(Ngrid, dtype="float")
    nobj_in_cell = np.zeros(Ngrid, dtype="int")

    for i in range(Nobj):
        dX = (xgrid - xobj[i]) / grid_size
        dY = (ygrid - yobj[i]) / grid_size
        dZ = (zgrid - zobj[i]) / grid_size

        w = weight_fun(dX) * weight_fun(dY) * weight_fun(dZ)

        grid_val += w
        grid_var += w**2

        nobj_in_cell += (w != 0).astype("int")

    return grid_val, grid_var, nobj_in_cell


def define_randoms(
    random_method,
    xobj,
    yobj,
    zobj,
    raobj,
    decobj,
    rcomobj,
    Nrandom=None,
    coord_randoms=None,
    max_coordinates=None,
):
    """Generate random positions for density estimation.

    Supports cartesian uniform, choice-based on observed distributions, or from file.

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

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Random x, y, z positions.
    """
    N = xobj.size

    # Uniform in X,Y,Z
    if random_method == "cartesian":
        xobj_random = (np.max(xobj) - np.min(xobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(xobj)
        yobj_random = (np.max(yobj) - np.min(yobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(yobj)
        zobj_random = (np.max(zobj) - np.min(zobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(zobj)

    # Choice in the ra, dec, and redshift data coordinates to create the random.
    elif random_method == "choice":
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


def grid_data_density(
    grid,
    grid_size,
    ra,
    dec,
    rcom,
    kind="ngp",
    n_cut=None,
    weight_min=None,
    verbose=False,
    compute_density=True,
    Nrandom=10,
    random_method="cartesian",
    coord_randoms=None,
):
    """Grid objects onto a mesh and compute density contrast and errors.

    Args:
        grid (dict): Grid coordinates dictionary from constructors.
        grid_size (float): Cell size.
        ra (array-like): Object right ascensions.
        dec (array-like): Object declinations.
        rcom (array-like): Object comoving distances.
        kind (str): Resampler kind (`ngp`, `cic`, `tsc`, `pcs`, `ngp_errw`).
        n_cut (int, optional): Minimum objects per cell.
        weight_min (float, optional): Minimum weight per cell.
        verbose (bool): Print number of cells.
        compute_density (bool): Whether to compute density contrast.
        Nrandom (int): Randoms per object if computing density.
        random_method (str): Random generation method.
        coord_randoms (tuple, optional): Randoms coordinates for `file` method.

    Returns:
        dict: Updated grid with weights, counts, density, and errors.
    """
    # Check valid input grid kind
    kind = kind.lower()
    if kind not in _GRID_KIND:
        raise ValueError(
            "INVALID GRID TYPE ! \n Grid allowed : " + f"{k}" for k in _GRID_KIND
        )

    xobj, yobj, zobj = utils.radec2cart(rcom, ra, dec)
    xgrid, ygrid, zgrid = utils.radec2cart(grid["rcom_zobs"], grid["ra"], grid["dec"])

    # Compute weight
    weight_fun = globals()[kind + "_weight"]
    sum_weights, var_weights, n_in_cell = attribute_weight_density(
        grid_size, xobj, yobj, zobj, xgrid, ygrid, zgrid, weight_fun
    )

    grid["sum_weights"] = sum_weights
    grid["var_weights"] = var_weights
    grid["nincell"] = n_in_cell

    if compute_density:
        (
            xobj_random,
            yobj_random,
            zobj_random,
        ) = define_randoms(
            random_method,
            xobj,
            yobj,
            zobj,
            ra,
            dec,
            rcom,
            Nrandom=Nrandom,
            coord_randoms=coord_randoms,
        )

        (
            sum_weights_random,
            var_weights_random,
            n_in_cell_random,
        ) = attribute_weight_density(
            grid_size,
            xobj_random,
            yobj_random,
            zobj_random,
            xgrid,
            ygrid,
            zgrid,
            weight_fun,
        )
        grid["sum_weights_random"] = sum_weights_random
        grid["var_weights_random"] = var_weights_random
        grid["nincell_random"] = n_in_cell_random

        grid["density"] = (
            (sum_weights / np.sum(sum_weights))
            / (sum_weights_random / np.sum(sum_weights_random))
        ) - 1

        grid["density_error"] = np.sqrt(
            1 / (n_in_cell_random * (np.sum(n_in_cell) / np.sum(n_in_cell_random)))
        )

        grid["density_nincell"] = (
            (n_in_cell / np.sum(n_in_cell))
            / (n_in_cell_random / np.sum(n_in_cell_random))
        ) - 1

    cut_grid(grid, n_cut=n_cut, weight_min=weight_min)

    if verbose:
        print(f"N cells in grid = {len(grid['ra'])}")
    return grid


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


def grid_data_density_pypower(
    raobj,
    decobj,
    rcomobj,
    rcom_max,
    grid_size,
    grid_type,
    kind,
    Nrandom=10,
    random_method="cartesian",
    interlacing=2,
    compensate=False,
    coord_randoms=None,
    min_count_random=0,
    overhead=20,
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
    xobj, yobj, zobj = utils.radec2cart(rcomobj, raobj, decobj)
    mask = np.abs(xobj) < rcom_max + overhead
    mask &= np.abs(yobj) < rcom_max + overhead
    mask &= np.abs(zobj) < rcom_max + overhead
    xobj, yobj, zobj = xobj[mask], yobj[mask], zobj[mask]
    raobj, decobj, rcomobj = raobj[mask], decobj[mask], rcomobj[mask]

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
    )

    data_positions = np.array([xobj, yobj, zobj]).T
    randoms_positions = np.array([xobj_random, yobj_random, zobj_random]).T

    data_weights = np.ones((data_positions.shape[0],))
    randoms_weights = np.ones((randoms_positions.shape[0],))

    catalog_mesh = CatalogMesh(
        data_positions=data_positions,
        data_weights=data_weights,
        randoms_positions=randoms_positions,
        randoms_weights=randoms_weights,
        interlacing=interlacing,
        boxsize=2 * (rcom_max + overhead),
        boxcenter=0.0,
        cellsize=grid_size,
        resampler=kind,
        position_type="pos",
    )

    catalog_mesh_count = CatalogMesh(
        data_positions=data_positions,
        data_weights=data_weights,
        randoms_positions=randoms_positions,
        randoms_weights=randoms_weights,
        interlacing=0,
        boxsize=2 * (rcom_max + overhead),
        boxcenter=0.0,
        cellsize=grid_size,
        resampler="ngp",
        position_type="pos",
    )

    mesh_data = catalog_mesh.to_mesh(field="data", compensate=compensate)
    mesh_randoms = catalog_mesh.to_mesh(
        field="data-normalized_randoms", compensate=compensate
    )

    mesh_count_randoms = catalog_mesh_count.to_mesh(field="data-normalized_randoms")

    density_contrast = np.ravel(mesh_data.value / mesh_randoms.value - 1)

    count_randoms = np.ravel(mesh_count_randoms.value).astype(int)
    density_contrast_err = np.full(count_randoms.shape, np.nan)
    mask = count_randoms > min_count_random
    density_contrast_err[mask] = np.sqrt(1 / (count_randoms[mask]))

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

    return grid


def grid_data_velocity_pypower(
    raobj,
    decobj,
    rcomobj,
    rcom_max,
    variance,
    velocity,
    grid_size,
    grid_type,
    kind,
    interlacing=0,
    compensate=False,
    overhead=20,
):
    """Grid velocity catalog with pypower and compute weighted means and variance.

    Args:
        raobj (array-like): Right ascensions.
        decobj (array-like): Declinations.
        rcomobj (array-like): Comoving distances.
        rcom_max (float): Outer cutoff.
        variance (array-like): Per-object variances.
        velocity (array-like|None): Per-object velocities (optional for variance-only).
        grid_size (float): Cell size.
        grid_type (str): `rect` or `sphere` cut behavior.
        kind (str): Resampler passed to CatalogMesh.
        interlacing (int): Interlacing factor.
        compensate (bool): Apply resampler compensation.
        overhead (float): Extra margin around cutoff.

    Returns:
        dict: Grid with positions, velocity (optional), variance, and counts.
    """
    xobj, yobj, zobj = utils.radec2cart(rcomobj, raobj, decobj)
    mask = np.abs(xobj) < rcom_max + overhead
    mask &= np.abs(yobj) < rcom_max + overhead
    mask &= np.abs(zobj) < rcom_max + overhead
    xobj, yobj, zobj = xobj[mask], yobj[mask], zobj[mask]
    raobj, decobj, rcomobj = raobj[mask], decobj[mask], rcomobj[mask]

    data_positions = np.array([xobj, yobj, zobj]).T
    count_weights = np.ones((data_positions.shape[0],))

    catalog_mesh_var = CatalogMesh(
        data_positions=data_positions,
        data_weights=variance,
        interlacing=interlacing,
        boxsize=2 * (rcom_max + overhead),
        boxcenter=0.0,
        cellsize=grid_size,
        resampler=kind,
        position_type="pos",
    )
    catalog_mesh_count = CatalogMesh(
        data_positions=data_positions,
        data_weights=count_weights,
        interlacing=interlacing,
        boxsize=2 * (rcom_max + overhead),
        boxcenter=0.0,
        cellsize=grid_size,
        resampler=kind,
        position_type="pos",
    )
    if velocity is not None:
        weights_weighted_mean = velocity / variance
        catalog_mesh_weighted_vel = CatalogMesh(
            data_positions=data_positions,
            data_weights=weights_weighted_mean,
            interlacing=interlacing,
            boxsize=2 * (rcom_max + overhead),
            boxcenter=0.0,
            cellsize=grid_size,
            resampler=kind,
            position_type="pos",
        )
        catalog_mesh_invvar = CatalogMesh(
            data_positions=data_positions,
            data_weights=1 / variance,
            interlacing=interlacing,
            boxsize=2 * (rcom_max + overhead),
            boxcenter=0.0,
            cellsize=grid_size,
            resampler=kind,
            position_type="pos",
        )
    mesh_var = catalog_mesh_var.to_mesh(field="data", compensate=compensate)
    mesh_count = catalog_mesh_count.to_mesh(field="data")
    N_in_cell = np.ravel(mesh_count.value)
    variance_grid = np.ravel(mesh_var.value) / (
        N_in_cell**2
    )  # *N_in_cell/np.abs(N_in_cell)
    if velocity is not None:
        mesh_weighted_vel = catalog_mesh_weighted_vel.to_mesh(
            field="data", compensate=compensate
        )
        mesh_invvar = catalog_mesh_invvar.to_mesh(field="data", compensate=compensate)
        velocity_grid = np.ravel(mesh_weighted_vel.value) / np.ravel(mesh_invvar.value)
    else:
        velocity_grid = None

    coord_mesh = np.array(
        np.meshgrid(
            np.sort(mesh_var.x[0][:, 0, 0]),
            np.sort(mesh_var.x[1][0, :, 0]),
            np.sort(mesh_var.x[2][0, 0, :]),
            indexing="ij",
        )
    )
    xgrid = np.ravel(coord_mesh[0, :, :, :]) + grid_size / 2
    ygrid = np.ravel(coord_mesh[1, :, :, :]) + grid_size / 2
    zgrid = np.ravel(coord_mesh[2, :, :, :]) + grid_size / 2
    rcomgrid, ragrid, decgrid = utils.cart2radec(xgrid, ygrid, zgrid)
    if velocity is not None:
        grid = {
            "x": xgrid,
            "y": ygrid,
            "z": zgrid,
            "ra": ragrid,
            "dec": decgrid,
            "rcom_zobs": rcomgrid,
            "velocity": velocity_grid,
            "velocity_variance": variance_grid,
            "N_in_cell": N_in_cell,
        }
    else:
        grid = {
            "x": xgrid,
            "y": ygrid,
            "z": zgrid,
            "ra": ragrid,
            "dec": decgrid,
            "rcom_zobs": rcomgrid,
            "velocity_variance": variance_grid,
            "N_in_cell": N_in_cell,
        }
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


# CR - First try unconnecting pypower


# def _get_compensation_window(resampler="cic", shotnoise=False):
#     r"""
#     Return the compensation function, which corrects for the particle-mesh assignment (resampler) kernel.

#     Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/source/mesh/catalog.py,
#     following https://arxiv.org/abs/astro-ph/0409240.
#     ("shotnoise" formula for pcs has been checked with WolframAlpha).

#     Parameters
#     ----------
#     resampler : string, default='cic'
#         Resampler used to assign particles to the mesh.
#         Choices are ['ngp', 'cic', 'tcs', 'pcs'].

#     shotnoise : bool, default=False
#         If ``False``, return expression for eq. 18 in https://arxiv.org/abs/astro-ph/0409240.
#         This the correct choice when applying interlacing, as aliased images (:math:`\mathbf{n} \neq (0,0,0)`) are suppressed in eq. 17.
#         If ``True``, return expression for eq. 19.

#     Returns
#     -------
#     window : callable
#         Window function, taking as input :math:`\pi k_{i} / k_{N} = k / c`
#         where :math:`k_{N}` is the Nyquist wavenumber and :math:`c` is the cell size,
#         for each :math:`x`, :math:`y`, :math:`z`, axis.
#     """
#     resampler = resampler.lower()

#     if shotnoise:
#         if resampler == "ngp":

#             def window(*x):
#                 return 1.0

#         elif resampler == "cic":

#             def window(*x):
#                 toret = 1.0
#                 for xi in x:
#                     toret = toret * (1 - 2.0 / 3 * np.sin(0.5 * xi) ** 2) ** 0.5
#                 return toret

#         elif resampler == "tsc":

#             def window(*x):
#                 toret = 1.0
#                 for xi in x:
#                     s = np.sin(0.5 * xi) ** 2
#                     toret = toret * (1 - s + 2.0 / 15 * s**2) ** 0.5
#                 return toret

#         elif resampler == "pcs":

#             def window(*x):
#                 toret = 1.0
#                 for xi in x:
#                     s = np.sin(0.5 * xi) ** 2
#                     toret = (
#                         toret
#                         * (
#                             1
#                             - 4.0 / 3.0 * s
#                             + 2.0 / 5.0 * s**2
#                             - 4.0 / 315.0 * s**3
#                         )
#                         ** 0.5
#                     )
#                 return toret

#     else:
#         p = {"ngp": 1, "cic": 2, "tsc": 3, "pcs": 4}[resampler]

#         def window(*x):
#             toret = 1.0
#             for xi in x:
#                 toret = toret * np.sinc(0.5 / np.pi * xi) ** p
#             return toret

#     return window


# def _get_resampler(resampler):
#     conversions = {"ngp": "nnb", "cic": "cic", "tsc": "tsc", "pcs": "pcs"}
#     resampler = conversions[resampler]
#     from pmesh.window import FindResampler

#     return FindResampler(resampler)


# def paint(
#     positions, weights, out, resampler, scaling=None, transform=None, offset=None
# ):
#     from pmesh.pm import ParticleMesh

#     pm = ParticleMesh(BoxSize=2 * rcom_max, Nmesh=(10, 10, 10))

#     positions = positions - offset
#     factor = bool(interlacing) + 0.5
#     scalar_weights = weights is None

#     if scaling is not None:
#         if scalar_weights:
#             weights = scaling
#         else:
#             weights = weights * scaling

#     def paint_slab(sl):
#         # Decompose positions such that they live in the same region as the mesh in the current process
#         p = positions[sl]
#         size = len(p)
#         layout = pm.decompose(p, smoothing=factor * self.resampler.support)
#         p = layout.exchange(p)
#         w = weights if scalar_weights else layout.exchange(weights[sl])
#         # hold = True means no zeroing of out
#         pm.paint(
#             p, mass=w, resampler=resampler, transform=transform, hold=True, out=out
#         )
#         return size

#     islab = 0
#     slab_npoints_max = int(1024 * 1024 * 4)
#     slab_npoints = slab_npoints_max
#     sizes = pm.comm.allgather(len(positions))
#     local_size_max = max(sizes)
#     painted_size = 0

#     while islab < local_size_max:
#         sl = slice(islab, islab + slab_npoints)
#         painted_size_slab = paint_slab(sl)
#         painted_size += painted_size_slab
#         islab += slab_npoints
#         slab_npoints = min(slab_npoints_max, int(slab_npoints * 1.2))


# def compensate(self, cfield):
#     if self.mpicomm.rank == 0:
#         self.log_info("Applying compensation {}.".format(self.compensation))
#     # Apply compensation window for particle-assignment scheme
#     window = _get_compensation_window(**self.compensation)

#     cellsize = self.boxsize / self.nmesh
#     for k, slab in zip(cfield.slabs.x, cfield.slabs):
#         kc = tuple(ki * ci for ki, ci in zip(k, cellsize))
#         slab[...] /= window(*kc)


# def compute_density(positions, weights):
#     out = pm.create(type="real", value=0.0)
#     for p, w in zip(positions, weights):
#         paint(p, *w, out)

#     if compensate:
#         out = out.r2c()
#         compensate(out)
#         out = out.c2r()
#     return out
