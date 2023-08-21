import numpy as np
from flip.utils import create_log

log = create_log()
try:
    from pypower import CatalogMesh
except:
    log.add("No pypower module detected, gridding with this method is unavailable")

# CR - no cut in healpix implemented

_GRID_KIND = ["ngp", "ngp_errw", "cic", "tsc", "pcs"]


def _compute_grid_window(grid_size, k, order, n):
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
    _order_dic = {
        "ngp": 1,
        "ngp_errw": 1,
        "cic": 2,
        "tsc": 3,
        "pcs": 4,
    }

    if kind not in _GRID_KIND:
        raise ValueError(
            "INVALID GRID TYPE ! Grid allowed : "
            + "".join("" + f"{k} " for k in _GRID_KIND)
        )
    if grid_size == 0:
        return None
    return _compute_grid_window(grid_size, kh, _order_dic[kind], n)


def radec2cart(rcom, ra, dec):
    x = rcom * np.cos(ra) * np.cos(dec)
    y = rcom * np.sin(ra) * np.cos(dec)
    z = rcom * np.sin(dec)
    return x, y, z


def construct_grid_regular_sphere(grid_size, rcom_max):
    """Construct a regular spherical grid"""

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
    center_r_comov = np.sqrt(cp_x**2 + cp_y**2 + cp_z**2)
    center_ra = np.arctan2(cp_y, cp_x)
    mask = center_r_comov != 0
    center_dec = np.zeros(center_ra.shape)
    center_dec[mask] = np.arcsin(cp_z[mask] / center_r_comov[mask])

    # Cut the grid with rcom_max
    grid = {
        "ra": center_ra,
        "dec": center_dec,
        "rcom": center_r_comov,
        "x": cp_x,
        "y": cp_y,
        "z": cp_z,
    }
    cut_grid(grid, n_cut=None, weight_min=None, rcom_max=rcom_max)

    return grid


def construct_grid_regular_rectangular(grid_size, rcom_max):
    """Construct a regular rectangular grid"""

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
    center_r_comov = np.sqrt(cp_x**2 + cp_y**2 + cp_z**2)
    center_ra = np.arctan2(cp_y, cp_x)
    mask = center_r_comov != 0
    center_dec = np.zeros(center_ra.shape)
    center_dec[mask] = np.arcsin(cp_z[mask] / center_r_comov[mask])

    # Cut the grid with rcom_max
    grid = {
        "ra": center_ra,
        "dec": center_dec,
        "rcom": center_r_comov,
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
    """Clood In Cell."""
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
    grid_size, xobj, yobj, zobj, xgrid, ygrid, zgrid, weight_fun
):
    """Attribute weighted value and errors to cells.

    Parameters
    ----------
    grid_size : float
        _description_
    xobj : array(float)
        _description_
    yobj : array(float)
        _description_
    zobj : array(float)
        _description_
    xgrid : array(float)
        _description_
    ygrid : array(float)
        _description_
    zgrid : array(float)
        _description_
    weight_fun : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
    Nrandom=100,
    random_method="cartesian",
):
    """Grid the velocities with the given grid_size and method.

    Parameters
    ----------
    grid_size : float
        Grid size in Mpc/h
    ra : array(float)
        Obj RA coord
    dec : array(float)
        Obj Dec corrd
    rcom : array(float)
        Obj comoving distance in Mpc/h
    kind : str, optional
        THe method used to compute voxcell values, by default 'ngp'
    n_cut : int, optional
        Minimum number of obj that contribute to a voxcell, by default no cut
    weight_min : int, optional
        Minimum weight in a voxcell, by default no cut
    Returns
    -------
    dict
        grid properties
    """
    # Check valid input grid kind
    kind = kind.lower()
    if kind not in _GRID_KIND:
        raise ValueError(
            "INVALID GRID TYPE ! \n Grid allowed : " + f"{k}" for k in _GRID_KIND
        )

    xobj, yobj, zobj = radec2cart(rcom, ra, dec)
    xgrid, ygrid, zgrid = radec2cart(grid["rcom"], grid["ra"], grid["dec"])

    # Compute weight
    weight_fun = globals()[kind + "_weight"]
    sum_weights, var_weights, n_in_cell = attribute_weight_density(
        grid_size, xobj, yobj, zobj, xgrid, ygrid, zgrid, weight_fun
    )

    grid["sum_weights"] = sum_weights
    grid["var_weights"] = var_weights
    grid["nincell"] = n_in_cell

    if compute_density:
        N = ra.size
        # Choice in the ra, dec, and redshift data coordinates to create the random.
        if random_method == "choice":
            counts_ra, bins_ra = np.histogram(ra, bins=1000)
            ra_random = np.random.choice(
                (bins_ra[1:] + bins_ra[:-1]) / 2,
                p=counts_ra / float(counts_ra.sum()),
                size=Nrandom * N,
            )
            counts_dec, bins_dec = np.histogram(dec, bins=1000)
            dec_random = np.random.choice(
                (bins_dec[1:] + bins_dec[:-1]) / 2,
                p=counts_dec / float(counts_dec.sum()),
                size=Nrandom * N,
            )
            counts_rcom, bins_rcom = np.histogram(rcom, bins=1000)
            rcom_random = np.random.choice(
                (bins_rcom[1:] + bins_rcom[:-1]) / 2,
                p=counts_rcom / float(counts_rcom.sum()),
                size=Nrandom * N,
            )

            xobj_random, yobj_random, zobj_random = radec2cart(
                rcom_random, ra_random, dec_random
            )

        # Uniform in ra and dec based on a healpix footprint. Choice in the redshift data coordinates.
        elif random_method == "healpix":
            ra_random = (np.max(ra) - np.min(ra)) * np.random.random(
                size=Nrandom * N
            ) + np.min(ra)
            dec_random = (np.max(dec) - np.min(dec)) * np.random.random(
                size=Nrandom * N
            ) + np.min(dec)
            counts_rcom, bins_rcom = np.histogram(rcom, bins=1000)
            rcom_random = np.random.choice(
                (bins_rcom[1:] + bins_rcom[:-1]) / 2,
                p=counts_rcom / float(counts_rcom.sum()),
                size=Nrandom * N,
            )

            xobj_random, yobj_random, zobj_random = radec2cart(
                rcom_random, ra_random, dec_random
            )

        # Uniform in X,Y,Z
        elif random_method == "cartesian":
            xobj_random = (np.max(xobj) - np.min(xobj)) * np.random.random(
                size=Nrandom * N
            ) + np.min(xobj)
            yobj_random = (np.max(yobj) - np.min(yobj)) * np.random.random(
                size=Nrandom * N
            ) + np.min(yobj)
            zobj_random = (np.max(zobj) - np.min(zobj)) * np.random.random(
                size=Nrandom * N
            ) + np.min(zobj)

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
        grid["density_err"] = np.sqrt(Nrandom / n_in_cell_random)

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
    n_cut=None,
    weight_min=None,
    rcom_max=None,
    xmax=None,
    ymax=None,
    zmax=None,
):
    """Cut low quality cells

    n_cut : int, optional
         Minimum number of obj that contribute to a voxcell, by default no cut
    weight_min : int, optional
         Minimum weight in a voxcell, by default no cut
    rcom_max : float, optional
        Maximal comoving distance of a voxcell, by default no cut"""

    mask = np.full(grid["ra"].shape, True)
    if n_cut is not None:
        mask &= grid["nincell"] > n_cut
    if weight_min is not None:
        mask &= grid["sum_weights"] > weight_min
    if rcom_max is not None:
        mask &= grid["rcom"] < rcom_max
    if xmax is not None:
        mask &= np.abs(grid["x"]) < xmax
    if ymax is not None:
        mask &= np.abs(grid["y"]) < ymax
    if zmax is not None:
        mask &= np.abs(grid["z"]) < zmax
    if remove_nan_density:
        if "density" in grid:
            mask &= ~(np.isnan(grid["density"]))
        if "density_err" in grid:
            mask &= ~(np.isnan(grid["density_err"]))

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
):
    xobj, yobj, zobj = radec2cart(rcomobj, raobj, decobj)
    mask = np.abs(xobj) < rcom_max
    mask &= np.abs(yobj) < rcom_max
    mask &= np.abs(zobj) < rcom_max
    xobj, yobj, zobj = xobj[mask], yobj[mask], zobj[mask]
    raobj, decobj, rcomobj = raobj[mask], decobj[mask], rcomobj[mask]

    N = raobj.size
    # Choice in the ra, dec, and redshift data coordinates to create the random.
    if random_method == "choice":
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

        xobj_random, yobj_random, zobj_random = radec2cart(
            rcom_random, ra_random, dec_random
        )

    # Uniform in ra and dec based on a healpix footprint. Choice in the redshift data coordinates.
    elif random_method == "healpix":
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

        xobj_random, yobj_random, zobj_random = radec2cart(
            rcom_random, ra_random, dec_random
        )

    # Uniform in X,Y,Z
    elif random_method == "cartesian":
        xobj_random = (np.max(xobj) - np.min(xobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(xobj)
        yobj_random = (np.max(yobj) - np.min(yobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(yobj)
        zobj_random = (np.max(zobj) - np.min(zobj)) * np.random.random(
            size=Nrandom * N
        ) + np.min(zobj)

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
        boxsize=2 * rcom_max,
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
        boxsize=2 * rcom_max,
        boxcenter=0.0,
        cellsize=grid_size,
        resampler="ngp",
        position_type="pos",
    )

    mesh = catalog_mesh.to_mesh(field="normalized_data", compensate=compensate)
    mesh_count = catalog_mesh_count.to_mesh(field="randoms", compensate=False) / Nrandom

    coord_mesh = np.array(
        np.meshgrid(
            np.sort(mesh.slabs.x.optx[0][:, 0, 0]),
            np.sort(mesh.slabs.x.optx[1][0, :, 0]),
            np.sort(mesh.slabs.x.optx[2][0, 0, :]),
            indexing="ij",
        )
    )
    xgrid = np.ravel(coord_mesh[0, :, :, :])
    ygrid = np.ravel(coord_mesh[1, :, :, :])
    zgrid = np.ravel(coord_mesh[2, :, :, :])

    rcomgrid = np.sqrt(xgrid**2 + ygrid**2 + zgrid**2)
    ragrid = np.arctan2(ygrid, xgrid)
    mask = rcomgrid != 0
    decgrid = np.zeros(ragrid.shape)
    decgrid[mask] = np.arcsin(zgrid[mask] / rcomgrid[mask])

    density_contrast = np.ravel(mesh.value - 1)

    count = np.ravel(mesh_count)
    density_contrast_err = np.full_like(count, np.nan)
    mask = count != 0.0
    density_contrast_err[mask] = np.sqrt(1 / count[mask])

    grid = {
        "x": xgrid,
        "y": ygrid,
        "z": zgrid,
        "ra": ragrid,
        "dec": decgrid,
        "rcom": rcomgrid,
        "density": density_contrast,
        "density_err": density_contrast_err,
    }

    if grid_type == "rect":
        cut_grid(
            grid, remove_nan_density=True, xmax=rcom_max, ymax=rcom_max, zmax=rcom_max
        )

    if grid_type == "sphere":
        cut_grid(grid, remove_nan_density=True, rcom_max=rcom_max)

    return grid
