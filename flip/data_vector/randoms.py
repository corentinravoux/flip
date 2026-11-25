"""Random catalogues for density gridding.

Point randoms (uniform Cartesian, drawn from the data sky and distance
distributions, or read from a file) and kernel randoms, which copy data kernels
(e.g. GW localisation posteriors) and move them onto random positions.
`compute_random_counts_analytic` gives the expected counts per cell directly,
without randoms (`random_method="analytic"` in the gridding functions).
"""

import itertools
import time

import numpy as np

from flip._config import __use_jax__

if __use_jax__:
    try:
        import jax
        import jax.numpy as jnp

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:
    import numpy as jnp

    jax_installed = False

from flip import utils

_random_methods_avail = [
    "cartesian",
    "cartesian_max_coordinates",
    "choice",
    "choice_redshift",
    "choice_kernel",
    "file",
]


def _sky_kernels_to_cartesian(sky_kernels):
    """Convert sky kernels `(ra, dec, rcom)` to Cartesian, one kernel at a time.

    Uses numpy: jax would recompile for every distinct kernel size.

    Args:
        sky_kernels (list or array-like): Kernels of shape `(3, n_i)`; sizes
            `n_i` may differ between kernels.

    Returns:
        list[numpy.ndarray]: One `(n_i, 3)` array per kernel.
    """
    kernels_cartesian = []
    for kernel in sky_kernels:
        ra, dec, rcom = np.asarray(kernel)
        cos_dec = np.cos(dec)
        kernels_cartesian.append(
            np.stack(
                [
                    rcom * cos_dec * np.cos(ra),
                    rcom * cos_dec * np.sin(ra),
                    rcom * np.sin(dec),
                ],
                axis=-1,
            )
        )
    return kernels_cartesian


def _to_kernel_frame(ra, dec, rcom, center):
    """Samples in their kernel frame: unit vector rotated by `-ra_center` about z,
    and distance relative to the kernel's mean distance."""
    ra_center, _, rcom_center = center
    cos_dec = np.cos(dec)
    return (
        cos_dec * np.cos(ra - ra_center),
        cos_dec * np.sin(ra - ra_center),
        np.sin(dec),
        rcom / rcom_center,
    )


def _from_kernel_frame(x, y, z, r_relative, dec_center, target):
    """Rotate kernel-frame samples about y by `dec_center - dec_target`, about z by
    `ra_target`, and scale distances by `rcom_target`. Returns Cartesian x, y, z."""
    ra_target, dec_target, rcom_target = target
    angle = dec_center - dec_target
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    x, z = x * cos_angle + z * sin_angle, z * cos_angle - x * sin_angle
    r = r_relative * rcom_target
    cos_ra, sin_ra = np.cos(ra_target), np.sin(ra_target)
    return r * (x * cos_ra - y * sin_ra), r * (x * sin_ra + y * cos_ra), r * z


def move_kernels(ra, dec, rcom, center, target):
    """Move kernel samples so that their kernel centre lands on a target.

    Each kernel is rigidly rotated on the sphere so that its mean direction
    points to the target direction: rotation about z by `-ra_center`, about y
    by `dec_center - dec_target`, then about z by `ra_target`. Its comoving
    distances are rescaled by `rcom_target / rcom_center`. The angular shape
    and the relative distance spread of each kernel are preserved.

    Works on flat samples: every argument has one entry per sample, so kernels
    of different sizes are moved in one vectorised call (numpy).

    Args:
        ra (numpy.ndarray): Sample right ascensions in radians.
        dec (numpy.ndarray): Sample declinations in radians.
        rcom (numpy.ndarray): Sample comoving distances.
        center (tuple): `(ra_center, dec_center, rcom_center)` of the sample's
            kernel, one entry per sample.
        target (tuple): `(ra_target, dec_target, rcom_target)` for the sample's
            kernel, one entry per sample.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Moved x, y, z.
    """
    x, y, z, r_relative = _to_kernel_frame(ra, dec, rcom, center)
    return _from_kernel_frame(x, y, z, r_relative, center[1], target)


def _kernel_centers(ra, dec, rcom, offsets, sizes):
    """Mean direction and mean distance of each kernel from flat samples."""
    cos_dec = np.cos(dec)
    x = np.add.reduceat(cos_dec * np.cos(ra), offsets) / sizes
    y = np.add.reduceat(cos_dec * np.sin(ra), offsets) / sizes
    z = np.add.reduceat(np.sin(dec), offsets) / sizes
    ra_center = np.arctan2(y, x)
    dec_center = np.arctan2(z, np.hypot(x, y))
    rcom_center = np.add.reduceat(rcom, offsets) / sizes
    return ra_center, dec_center, rcom_center


def _init_random_key(seed=None):
    """Seed the generator; return a JAX key, or `None` with numpy."""
    if seed is None:
        if jax_installed:
            return jax.random.key(int(time.time() * 1e6))
        return None
    if jax_installed:
        return jax.random.key(seed)
    jnp.random.seed(seed)
    return None


def _uniform(key, bounds, size):
    """Draw one uniform sample of length `size` per `(low, high)` in `bounds`.

    Returns:
        tuple: Updated key and list of draws.
    """
    if jax_installed:
        key, *subkeys = jax.random.split(key, num=len(bounds) + 1)
        draws = [
            (high - low) * jax.random.uniform(k, shape=(size,)) + low
            for k, (low, high) in zip(subkeys, bounds)
        ]
    else:
        draws = [
            (high - low) * jnp.random.random(size=size) + low for low, high in bounds
        ]
    return key, draws


def _choice_histogram(key, values, size, bins=1000):
    """Draw `size` values from the histogram of `values` (bin centres).

    Returns:
        tuple: Updated key and draws.
    """
    counts, edges = jnp.histogram(values, bins=bins)
    centres = (edges[1:] + edges[:-1]) / 2
    p = counts / float(counts.sum())
    if jax_installed:
        key, k = jax.random.split(key)
        return key, jax.random.choice(k, centres, p=p, shape=(size,))
    return key, jnp.random.choice(centres, p=p, size=size)


def _uniform_sky(key, ra_min, ra_max, dec_min, dec_max, size):
    """Uniform per solid angle: uniform in ra and in sin(dec).

    Returns:
        tuple: Updated key, ra and dec draws.
    """
    key, (ra, sin_dec) = _uniform(
        key, [(ra_min, ra_max), (jnp.sin(dec_min), jnp.sin(dec_max))], size
    )
    return key, ra, jnp.arcsin(sin_dec)


def randoms_cartesian(key, xobj, yobj, zobj, Nrandom):
    """Uniform in the bounding box of the data, `Nrandom` per object."""
    _, (x, y, z) = _uniform(
        key,
        [(jnp.min(c), jnp.max(c)) for c in (xobj, yobj, zobj)],
        Nrandom * xobj.size,
    )
    return x, y, z


def randoms_cartesian_max_coordinates(key, max_coordinates, Nrandom):
    """Uniform in the cube `[-max_coordinates, max_coordinates]^3`, `Nrandom` in total."""
    _, (x, y, z) = _uniform(key, [(-max_coordinates, max_coordinates)] * 3, Nrandom)
    return x, y, z


def randoms_choice(key, raobj, decobj, rcomobj, Nrandom):
    """Draw ra, dec and rcom independently from the data histograms."""
    size = Nrandom * raobj.size
    key, ra = _choice_histogram(key, raobj, size)
    key, dec = _choice_histogram(key, decobj, size)
    _, rcom = _choice_histogram(key, rcomobj, size)
    return utils.spherical_to_cartesian(ra, dec, rcom)


def randoms_choice_redshift(key, raobj, decobj, rcomobj, Nrandom):
    """Uniform on the sky within the data ra/dec range; rcom from the data histogram."""
    size = Nrandom * raobj.size
    key, ra, dec = _uniform_sky(
        key, jnp.min(raobj), jnp.max(raobj), jnp.min(decobj), jnp.max(decobj), size
    )
    _, rcom = _choice_histogram(key, rcomobj, size)
    return utils.spherical_to_cartesian(ra, dec, rcom)


def randoms_choice_kernel(
    key, data_position_sky_kernel, Nrandom, d_rcom=10.0, chunk_size=20_000_000
):
    """Kernel randoms: copies of data kernels moved onto random positions.

    Random centres are uniform on the sky (within the kernels' ra/dec range),
    with rcom drawn from the histogram of the kernels' mean rcom. Each random
    takes a kernel chosen uniformly among those whose mean rcom is within
    `d_rcom`, moved with `move_kernels`. Kernels may have different sizes.
    The moves are vectorised over flat samples, in chunks of about
    `chunk_size` samples to bound memory.

    Args:
        key: JAX key (or `None` with numpy).
        data_position_sky_kernel (list or array-like): Kernels of shape `(3, n_i)`
            with rows `(ra, dec, rcom)`, angles in radians.
        Nrandom (int): Number of random kernels per data kernel.
        d_rcom (float): Half-width of the rcom window used to choose kernels.
        chunk_size (int): Approximate number of samples moved at once.

    Returns:
        tuple: x, y, z of the random centres, and a generator over chunks of
        random kernels. Each chunk is `(positions, sizes, start)`: Cartesian
        samples of shape `(S_chunk, 3)`, kernel after kernel, the number of
        samples of each random kernel in the chunk, and the index of the
        chunk's first random kernel.
    """
    N = len(data_position_sky_kernel)
    size = Nrandom * N
    sizes = np.array([np.shape(kernel)[1] for kernel in data_position_sky_kernel])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    ra_all, dec_all, rcom_all = np.concatenate(
        [np.asarray(kernel) for kernel in data_position_sky_kernel], axis=1
    )
    ra_center, dec_center, rcom_center = _kernel_centers(
        ra_all, dec_all, rcom_all, offsets, sizes
    )

    # RA wrapped to [0, 2 pi): otherwise samples outside it widen the range
    # beyond 2 pi and the overlap is drawn twice
    ra_wrapped = np.mod(ra_all, 2 * np.pi)
    key, ra_random, dec_random = _uniform_sky(
        key, ra_wrapped.min(), ra_wrapped.max(), dec_all.min(), dec_all.max(), size
    )
    key, rcom_random = _choice_histogram(key, jnp.asarray(rcom_center), size)
    _, (u,) = _uniform(key, [(0.0, 1.0)], size)
    ra_random, dec_random, rcom_random, u = (
        np.asarray(a) for a in (ra_random, dec_random, rcom_random, u)
    )

    # Uniform choice among kernels with |<rcom> - rcom_random| <= d_rcom
    order = np.argsort(rcom_center)
    r_sorted = rcom_center[order]
    lo = np.searchsorted(r_sorted, rcom_random - d_rcom, side="left")
    hi = np.searchsorted(r_sorted, rcom_random + d_rcom, side="right")
    if np.any(hi == lo):
        raise ValueError("No kernel within d_rcom of a random rcom: increase d_rcom")
    index_kernel = order[lo + np.floor(u * (hi - lo)).astype(int)]

    # Data samples in their kernel frame, computed once and reused by every copy
    kernel_of_data_sample = np.repeat(np.arange(N), sizes)
    x_frame, y_frame, z_frame, r_frame = _to_kernel_frame(
        ra_all,
        dec_all,
        rcom_all,
        (ra_center[kernel_of_data_sample], None, rcom_center[kernel_of_data_sample]),
    )

    # Per random kernel: rotation angles (only sin/cos per kernel, not per sample)
    angle = dec_center[index_kernel] - dec_random
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    cos_ra, sin_ra = np.cos(ra_random), np.sin(ra_random)

    # Move the chosen kernels chunk by chunk, vectorised over samples. Chunks are
    # produced lazily so that all random samples never sit in memory at once.
    sizes_random = sizes[index_kernel]
    n_per_chunk = max(1, int(chunk_size // sizes.mean()))

    def iterate_chunks():
        for start in range(0, size, n_per_chunk):
            stop = min(start + n_per_chunk, size)
            index_chunk = index_kernel[start:stop]
            sizes_chunk = sizes_random[start:stop]
            # sample -> random kernel, and sample -> data sample index
            kernel_of_sample = np.repeat(np.arange(start, stop), sizes_chunk)
            rank_in_kernel = np.arange(sizes_chunk.sum()) - np.repeat(
                np.cumsum(sizes_chunk) - sizes_chunk, sizes_chunk
            )
            sample = np.repeat(offsets[index_chunk], sizes_chunk) + rank_in_kernel

            x, y, z = x_frame[sample], y_frame[sample], z_frame[sample]
            ca, sa = cos_angle[kernel_of_sample], sin_angle[kernel_of_sample]
            x, z = x * ca + z * sa, z * ca - x * sa
            r = r_frame[sample] * rcom_random[kernel_of_sample]
            cr, sr = cos_ra[kernel_of_sample], sin_ra[kernel_of_sample]

            positions = np.empty((len(x), 3))
            positions[:, 0] = r * (x * cr - y * sr)
            positions[:, 1] = r * (x * sr + y * cr)
            positions[:, 2] = r * z
            yield positions, sizes_chunk, start

    x, y, z = utils.spherical_to_cartesian(ra_random, dec_random, rcom_random)
    return x, y, z, iterate_chunks()


def randoms_file(coord_randoms, max_coordinates=None):
    """Randoms from a `(N, 3)` array of `(ra, dec, rcom)`, optionally cut to a cube.

    Returns:
        tuple: x, y, z and the mask applied (`None` without `max_coordinates`).
    """
    x, y, z = utils.spherical_to_cartesian(
        coord_randoms[:, 0], coord_randoms[:, 1], coord_randoms[:, 2]
    )
    if max_coordinates is None:
        return x, y, z, None
    mask = jnp.abs(x) < max_coordinates
    mask &= jnp.abs(y) < max_coordinates
    mask &= jnp.abs(z) < max_coordinates
    return x[mask], y[mask], z[mask], mask


def _footprint_fraction_in_box(
    rcom_edges, sin_dec_edges, ra_min, ra_max, box_half_size, n_directions=512, n_sub=8
):
    """Fraction of each `(rcom, sin dec)` bin of the footprint inside a centred cube.

    Directions on a regular `(ra, sin dec)` grid of each bin, which is uniform
    per solid angle; `n_sub` distances per bin, weighted by `rcom**2`.

    Args:
        rcom_edges (numpy.ndarray): Edges of the rcom bins.
        sin_dec_edges (numpy.ndarray): Edges of the sin(dec) bins.
        ra_min, ra_max (float): RA range of the footprint, in radians.
        box_half_size (float): Half side of the cube.
        n_directions (int): Directions per side of each sin(dec) bin.
        n_sub (int): Distances per rcom bin.

    Returns:
        numpy.ndarray: Fraction of shape `(n_rcom_bins, n_sin_dec_bins)`.
    """
    t = (np.arange(n_directions) + 0.5) / n_directions
    ra = ra_min + (ra_max - ra_min) * t
    u = (np.arange(n_sub) + 0.5) / n_sub
    rcom_sub = rcom_edges[:-1, None] + np.diff(rcom_edges)[:, None] * u[None, :]

    fraction = np.empty((len(rcom_edges) - 1, len(sin_dec_edges) - 1))
    for i in range(len(sin_dec_edges) - 1):
        sin_dec = sin_dec_edges[i] + (sin_dec_edges[i + 1] - sin_dec_edges[i]) * t
        sin_dec, ra_direction = np.meshgrid(sin_dec, ra)
        cos_dec = np.sqrt(1.0 - sin_dec**2)
        max_abs = np.maximum.reduce(
            [
                np.abs(cos_dec * np.cos(ra_direction)),
                np.abs(cos_dec * np.sin(ra_direction)),
                np.abs(sin_dec),
            ]
        )
        # Distance at which each direction leaves the cube
        rcom_exit = np.sort(np.ravel(box_half_size / max_abs))
        n_inside = rcom_exit.size - np.searchsorted(rcom_exit, rcom_sub, side="right")
        fraction[:, i] = np.sum(rcom_sub**2 * n_inside, axis=1) / (
            np.sum(rcom_sub**2, axis=1) * rcom_exit.size
        )
    return fraction


def _nbar_at_positions(
    nbar, rcom_edges, sin_dec_edges, ra_min, ra_max, box_half_size, x, y, z
):
    """Expected density at Cartesian positions; 0 outside the bins, the footprint
    or the cube."""
    ra, dec, rcom = utils.cartesian_to_spherical(x, y, z)
    ra = np.mod(ra, 2 * np.pi)
    i_rcom = np.searchsorted(rcom_edges, rcom, side="right") - 1
    i_dec = np.searchsorted(sin_dec_edges, np.sin(dec), side="right") - 1
    inside = (i_rcom >= 0) & (i_rcom < len(rcom_edges) - 1)
    inside &= (i_dec >= 0) & (i_dec < len(sin_dec_edges) - 1)
    inside &= (ra >= ra_min) & (ra <= ra_max)
    inside &= np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)]) < box_half_size
    i_rcom = np.clip(i_rcom, 0, nbar.shape[0] - 1)
    i_dec = np.clip(i_dec, 0, nbar.shape[1] - 1)
    return np.where(inside, nbar[i_rcom, i_dec], 0.0)


def compute_random_counts_analytic(
    ra,
    dec,
    rcom,
    weights,
    xgrid,
    ygrid,
    zgrid,
    grid_size,
    box_half_size,
    d_rcom=10.0,
    n_bins_sin_dec=1,
    n_sub_cell=None,
):
    """Expected number of data objects per cell, computed without randoms.

    Analytic counterpart of the `choice_redshift` and `choice_kernel` randoms:
    their limit for `Nrandom -> infinity`. The expected density is taken from
    the data, uniform in ra within the data footprint (ra and dec ranges):

        nbar(rcom, sin dec) = S(rcom, sin dec) / V(rcom, sin dec),

    with `S` the weighted histogram of the data in bins of width `d_rcom` and
    `n_bins_sin_dec` bins of sin(dec), and `V` the volume of each bin inside
    the footprint and the cube `|x|, |y|, |z| < box_half_size`. Data outside
    the cube are ignored: mesh painting is periodic, so they must be cut
    before gridding. The count in a cell is `grid_size**3` times `nbar`
    averaged over `n_sub_cell**3` points of the cell, since coarse cells are
    wider than the radial structure of `nbar`.

    Args:
        ra (array-like): Data right ascensions in radians (kernel samples for
            kernel data).
        dec (array-like): Data declinations in radians.
        rcom (array-like): Data comoving distances.
        weights (array-like): Weight of each object or kernel sample; their
            sum is the number of objects (kernel samples weigh `1 / n_kernel`).
        xgrid, ygrid, zgrid (array-like): Cartesian cell centres.
        grid_size (float): Cell size.
        box_half_size (float): Half side of the cube holding the data.
        d_rcom (float): Width of the rcom bins. Point data give a noisier
            histogram than kernels: use a few `grid_size`.
        n_bins_sin_dec (int): Number of sin(dec) bins; 1 gives an expected
            density that depends on rcom only.
        n_sub_cell (int, optional): Points per cell side used to average
            `nbar`. `None` evaluates about 1e8 points in total (1 to 8 per side).

    Returns:
        numpy.ndarray: Expected number of objects per cell, the analytic
        equivalent of the scaled random counts.
    """
    ra, dec, rcom, weights = (np.asarray(a) for a in (ra, dec, rcom, weights))
    x, y, z = utils.spherical_to_cartesian(ra, dec, rcom)
    mask = np.maximum.reduce([np.abs(x), np.abs(y), np.abs(z)]) < box_half_size
    ra, dec, rcom, weights = ra[mask], dec[mask], rcom[mask], weights[mask]
    ra = np.mod(ra, 2 * np.pi)
    ra_min, ra_max = ra.min(), ra.max()

    rcom_edges = np.arange(0.0, rcom.max() + d_rcom, d_rcom)
    sin_dec_edges = np.linspace(
        np.sin(dec.min()), np.sin(dec.max()), n_bins_sin_dec + 1
    )
    histogram, _, _ = np.histogram2d(
        rcom, np.sin(dec), bins=[rcom_edges, sin_dec_edges], weights=weights
    )
    # Bin volume: integral of rcom**2 drcom dra dsin(dec), times the cube fraction
    volume = (
        (np.diff(rcom_edges**3) / 3)[:, None]
        * np.diff(sin_dec_edges)[None, :]
        * (ra_max - ra_min)
        * _footprint_fraction_in_box(
            rcom_edges, sin_dec_edges, ra_min, ra_max, box_half_size
        )
    )
    nbar = np.divide(histogram, volume, out=np.zeros_like(histogram), where=volume > 0)

    xgrid, ygrid, zgrid = (np.asarray(a) for a in (xgrid, ygrid, zgrid))
    if n_sub_cell is None:
        n_sub_cell = int(np.clip((1e8 / xgrid.size) ** (1 / 3), 1, 8))
    offsets = ((np.arange(n_sub_cell) + 0.5) / n_sub_cell - 0.5) * grid_size
    counts = np.zeros(xgrid.shape)
    for dx, dy, dz in itertools.product(offsets, repeat=3):
        counts += _nbar_at_positions(
            nbar,
            rcom_edges,
            sin_dec_edges,
            ra_min,
            ra_max,
            box_half_size,
            xgrid + dx,
            ygrid + dy,
            zgrid + dz,
        )
    return counts * grid_size**3 / n_sub_cell**3


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
    data_position_sky_kernel=None,
    seed=None,
    d_rcom=10.0,
):
    """Generate random Cartesian positions for density gridding.

    Dispatches to one `randoms_<method>` function per method.

    Args:
        random_method (str): `cartesian`, `cartesian_max_coordinates`, `choice`,
            `choice_redshift`, `choice_kernel`, or `file`.
        xobj, yobj, zobj (array-like): Data Cartesian positions (`cartesian`).
        raobj, decobj, rcomobj (array-like): Data sky positions (`choice`,
            `choice_redshift`).
        Nrandom (int): Number of randoms per object; total number for
            `cartesian_max_coordinates`.
        coord_randoms (array-like, optional): `(N, 3)` `(ra, dec, rcom)` for `file`.
        max_coordinates (float, optional): Half-size of the cube
            (`cartesian_max_coordinates`, cut for `file`).
        data_position_sky_kernel (list or array-like, optional): Data kernels
            (`choice_kernel`).
        seed (int, optional): Random seed used for reproducible draws.
        d_rcom (float): rcom window for the kernel choice (`choice_kernel`).

    Returns:
        tuple: Random x, y, z, the `file` mask (or `None`), and the random
        kernels (`choice_kernel`, else `None`).
    """

    def _needed_params(**needed_parameter):
        for name, p in needed_parameter.items():
            if p is None:
                raise ValueError(
                    f"random_method={random_method} requires {name} to be provided"
                )

    if random_method not in _random_methods_avail:
        allowed = ", ".join(_random_methods_avail)
        raise ValueError(f"INVALID random_method! Allowed: {allowed}")

    key = _init_random_key(seed)
    mask_random, kernel_randoms = None, None

    if random_method == "cartesian":
        _needed_params(xobj=xobj, yobj=yobj, zobj=zobj)
        x, y, z = randoms_cartesian(key, xobj, yobj, zobj, Nrandom)
    elif random_method == "cartesian_max_coordinates":
        _needed_params(max_coordinates=max_coordinates)
        x, y, z = randoms_cartesian_max_coordinates(key, max_coordinates, Nrandom)
    elif random_method == "choice":
        _needed_params(raobj=raobj, decobj=decobj, rcomobj=rcomobj)
        x, y, z = randoms_choice(key, raobj, decobj, rcomobj, Nrandom)
    elif random_method == "choice_redshift":
        _needed_params(raobj=raobj, decobj=decobj, rcomobj=rcomobj)
        x, y, z = randoms_choice_redshift(key, raobj, decobj, rcomobj, Nrandom)
    elif random_method == "choice_kernel":
        _needed_params(data_position_sky_kernel=data_position_sky_kernel)
        x, y, z, kernel_randoms = randoms_choice_kernel(
            key, data_position_sky_kernel, Nrandom, d_rcom=d_rcom
        )
    elif random_method == "file":
        _needed_params(coord_randoms=coord_randoms)
        x, y, z, mask_random = randoms_file(coord_randoms, max_coordinates)

    return x, y, z, mask_random, kernel_randoms
