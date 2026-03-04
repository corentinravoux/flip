try:
    from pmesh.pm import ParticleMesh
except ImportError:
    ParticleMesh = None
import numpy as np

from flip import utils

_grid_kind_avail = ["ngp", "ngp_errw", "cic", "tsc", "pcs"]


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


def _get_resampler(resampler):
    conversions = {"ngp": "nnb", "cic": "cic", "tsc": "tsc", "pcs": "pcs"}
    if resampler not in conversions:
        raise ValueError(
            "Unknown resampler {}, choices are {}".format(
                resampler, list(conversions.keys())
            )
        )
    resampler = conversions[resampler]
    from pmesh.window import FindResampler

    return FindResampler(resampler)


def _get_mesh_attrs(
    boxsize,
    cellsize,
):
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


def create_mesh(
    positions,
    boxsize,
    cellsize,
    assignement="ngp",
    weights=None,
    scaling=None,
):
    resampler = _get_resampler(assignement)

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
            layout = pm.decompose(p, smoothing=factor * resampler.support)
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
    for p, w in zip(positions, weights):
        paint(p, w, scaling, out)
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

    # Check valid input grid kind
    kind = kind.lower()
    if kind not in _grid_kind_avail:
        raise ValueError(
            "INVALID GRID TYPE ! \n Grid allowed : " + f"{k}" for k in _grid_kind_avail
        )

    data_positions, randoms_positions = prepare_data_position(
        data_position_sky,
        rcom_max,
        overhead,
        random_method,
        Nrandom,
        coord_randoms,
    )

    data_weights = np.ones((data_positions.shape[0],))
    randoms_weights = np.ones((randoms_positions.shape[0],))

    mesh_data = create_mesh(
        [data_positions],
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=[data_weights],
    )

    mesh_randoms = create_mesh(
        [randoms_positions],
        2 * (rcom_max + overhead),
        grid_size,
        assignement=kind,
        weights=[randoms_weights],
        scaling=np.sum(data_weights) / np.sum(randoms_weights),
    )

    mesh_count_randoms = create_mesh(
        [randoms_positions],
        2 * (rcom_max + overhead),
        grid_size,
        assignement="ngp",
        weights=[randoms_weights],
        scaling=np.sum(data_weights) / np.sum(randoms_weights),
    )

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


def prepare_data_position(
    data_position_sky,
    rcom_max,
    overhead,
    random_method,
    Nrandom,
    coord_randoms,
):

    rcomobj = data_position_sky[:, 0]
    raobj = data_position_sky[:, 1]
    decobj = data_position_sky[:, 2]

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

    return data_positions, randoms_positions
