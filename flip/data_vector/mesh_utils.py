"""Particle-mesh painting and readout in JAX, compatible with pmesh.

Single-process reimplementation of the parts of pmesh
(https://github.com/rainwoodman/pmesh) needed to grid catalogs:
mass-assignment windows, periodic painting and readout, the pmesh affine
convention, and ``RealField`` / ``ComplexField`` objects with pmesh
coordinates and FFT normalisation. No MPI domain decomposition.

Conventions (identical to pmesh):

- A position ``pos`` maps to grid units as ``g = pos * scale + translate``
  (``scale = Nmesh / BoxSize``). Mesh node ``i`` sits at ``g = i``.
- Windows are pmesh's tuned kernels. The stencil starts at
  ``floor(g + 1/2)`` for ``nnb``, ``floor(g)`` for ``cic``,
  ``floor(g + 1/2) - 1`` for ``tsc`` and ``floor(g) - 1`` for ``pcs``.
- Node indices wrap with the period (``Nmesh`` by default). Nodes outside
  the mesh after wrapping are dropped.
- Painting conserves mass: every node receives ``mass * W``, where ``W`` is
  the product of the 1D window weights. It is not divided by the cell volume.
- ``r2c`` is normalised by ``1 / prod(Nmesh)`` and ``c2r`` is not normalised.

One deliberate difference: the ``pcs`` window derivative (``gradient``)
includes the ``scale`` factor, as pmesh's generic ``cubic`` window does. The
pmesh tuned ``pcs`` kernel omits it.

Painting builds the ``support**ndim`` stencil of every particle with
separable 1D weights and accumulates it with a single flat scatter-add.
Particles are processed in fixed-size chunks under ``lax.scan``, which
bounds memory to ``chunk_size * support**ndim`` entries and keeps the loop
inside one compiled program. Everything is differentiable with respect to
positions and masses.

The same code runs on numpy, accumulating with ``np.bincount``, when JAX is
not installed or in a process forked after JAX started, where JAX would
deadlock.
"""

import math
import os

import numpy as np

from .._config import __use_jax__

if __use_jax__:
    try:
        import jax
        import jax.numpy as jnp

        jax_installed = True

    except ImportError:
        jax_installed = False
else:
    jax_installed = False


_resampler_aliases = {
    "nnb": "nnb",
    "ngp": "nnb",
    "nearest": "nnb",
    "cic": "cic",
    "linear": "cic",
    "tsc": "tsc",
    "quadratic": "tsc",
    "pcs": "pcs",
    "cubic": "pcs",
}

_resampler_support = {"nnb": 1, "cic": 2, "tsc": 3, "pcs": 4}

# Number of (particle, node) pairs handled per chunk. Each pair costs one
# index and one weight, so this is ~100 MB of temporaries in double precision.
_default_chunk_pairs = 2**23

_static_argnames = ("resampler", "period", "gradient", "chunk_size")

_in_forked_child = False


def _set_forked_child():
    global _in_forked_child
    _in_forked_child = True


if jax_installed and hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_set_forked_child)


def get_array_module(backend=None):
    """Return the array module used for mesh operations.

    Args:
        backend (str, optional): ``jax`` or ``numpy``. By default, JAX when it
            is installed, except in a forked child process (e.g. a
            ``multiprocessing`` fork pool) where JAX is not fork-safe.

    Returns:
        module: ``jax.numpy`` or ``numpy``.
    """
    if backend is None:
        backend = "jax" if jax_installed and not _in_forked_child else "numpy"
    if backend == "jax":
        if not jax_installed:
            raise ImportError("backend='jax' requires JAX to be installed")
        return jnp
    if backend == "numpy":
        return np
    raise ValueError(f"Unknown backend {backend}. Allowed: jax, numpy")


def get_resampler(resampler):
    """Return the canonical pmesh resampler name.

    Args:
        resampler (str): Resampler name. Aliases: ``ngp``/``nearest`` for
            ``nnb``, ``linear`` for ``cic``, ``quadratic`` for ``tsc``,
            ``cubic`` for ``pcs``.

    Returns:
        str: One of ``nnb``, ``cic``, ``tsc``, ``pcs``.
    """
    name = str(resampler).lower()
    if name not in _resampler_aliases:
        allowed = ", ".join(_resampler_aliases)
        raise ValueError(f"Unknown resampler {resampler}. Allowed: {allowed}")
    return _resampler_aliases[name]


def window_support(resampler):
    """Return the number of mesh nodes touched per axis by a resampler."""
    return _resampler_support[get_resampler(resampler)]


def _window_weights(g, resampler, derivative, xp):
    """Evaluate 1D window weights, as in pmesh ``_window_tuned_*.h``.

    Expressions keep the pmesh operation order, so that results agree to
    the last bit in most cases.

    Args:
        g (array): Positions in grid units.
        resampler (str): Canonical resampler name.
        derivative (bool): Return the derivative of the window with respect
            to ``g`` instead of the window itself.
        xp (module): Array module.

    Returns:
        tuple: Index of the first stencil node (float array) and the tuple of
        ``support`` weight arrays, one per stencil node.
    """
    if resampler == "nnb":
        left = xp.floor(g + 0.5)
        weight = xp.zeros_like(g) if derivative else xp.ones_like(g)
        return left, (weight,)

    if resampler == "cic":
        i0 = xp.floor(g)
        if derivative:
            return i0, (-xp.ones_like(g), xp.ones_like(g))
        v1 = g - i0
        return i0, (1.0 - v1, v1)

    if resampler == "tsc":
        i1 = xp.floor(g + 0.5)
        d0, d1, d2 = g - (i1 - 1), g - i1, g - (i1 + 1)
        if derivative:
            return i1 - 1, (-(1.5 - d0), -2 * d1, 1.5 + d2)
        return i1 - 1, (
            (1.5 - d0) * (1.5 - d0) * 0.5,
            0.75 - d1 * d1,
            (1.5 + d2) * (1.5 + d2) * 0.5,
        )

    if resampler == "pcs":
        i1 = xp.floor(g)
        d0, d1, d2, d3 = g - (i1 - 1), g - i1, g - (i1 + 1), g - (i1 + 2)
        if derivative:
            return i1 - 1, (
                -1.0 / 2.0 * (2 - d0) * (2 - d0),
                +1.0 / 6.0 * (-12 * d1 + 9 * d1 * d1),
                -1.0 / 6.0 * (+12 * d2 + 9 * d2 * d2),
                +1.0 / 2.0 * (2 + d3) * (2 + d3),
            )
        return i1 - 1, (
            1.0 / 6.0 * (2 - d0) * (2 - d0) * (2 - d0),
            1.0 / 6.0 * (4 - 6 * d1 * d1 + 3 * d1 * d1 * d1),
            1.0 / 6.0 * (4 - 6 * d2 * d2 - 3 * d2 * d2 * d2),
            1.0 / 6.0 * (2 + d3) * (2 + d3) * (2 + d3),
        )

    raise ValueError(f"Unknown resampler {resampler}")


def _stencil(positions, mass, shape, resampler, scale, translate, period, gradient, xp):
    """Build flat node indices and weights of every particle stencil.

    The mass multiplies the first-axis weights before the outer product, in
    the same operation order as pmesh.

    Args:
        positions (array): Positions with shape ``(N, ndim)``.
        mass (array or None): Masses with shape ``(N,)``, or ``None`` for
            unit masses.
        shape (tuple[int]): Mesh shape.
        resampler (str): Canonical resampler name.
        scale (array): Scale from position to grid units, per axis.
        translate (array): Translation in grid units, per axis.
        period (tuple[int]): Periodicity per axis, ``0`` for none.
        gradient (int or None): Axis along which the window is differentiated.
        xp (module): Array module.

    Returns:
        tuple: Flat indices and weights, both with shape
        ``(N, support**ndim)``.
    """
    ndim = len(shape)
    support = _resampler_support[resampler]
    index_dtype = xp.int32 if math.prod(shape) < 2**31 else xp.int64
    offsets = xp.arange(support, dtype=index_dtype)

    index, weight = None, None
    for d in range(ndim):
        g = positions[:, d] * scale[d] + translate[d]
        left, weights_d = _window_weights(g, resampler, gradient == d, xp)
        weights_d = xp.stack(weights_d, axis=-1)
        if gradient == d:
            weights_d = weights_d * scale[d]
        if d == 0 and mass is not None:
            weights_d = weights_d * mass[:, None]

        index_d = left.astype(index_dtype)[:, None] + offsets
        if period[d] > 0:
            index_d = xp.mod(index_d, period[d])
        if period[d] != shape[d]:
            inside = (index_d >= 0) & (index_d < shape[d])
            weights_d = xp.where(inside, weights_d, 0)
            index_d = xp.clip(index_d, 0, shape[d] - 1)

        if index is None:
            index, weight = index_d, weights_d
        else:
            new_axis = (-1,) + (1,) * d + (support,)
            index = index[..., None] * shape[d] + index_d.reshape(new_axis)
            weight = weight[..., None] * weights_d.reshape(new_axis)

    n = positions.shape[0]
    return index.reshape(n, -1), weight.reshape(n, -1)


def _get_chunk_size(chunk_size, n, resampler, ndim):
    """Return the number of particles processed per chunk."""
    if chunk_size is None:
        chunk_size = _default_chunk_pairs // _resampler_support[resampler] ** ndim
    return max(min(int(chunk_size), n), 1)


def _split_chunks(array, chunk_size):
    """Zero-pad the leading axis to a multiple of ``chunk_size`` and split it."""
    n_chunks = -(-array.shape[0] // chunk_size)
    padding = [(0, n_chunks * chunk_size - array.shape[0])]
    padding += [(0, 0)] * (array.ndim - 1)
    array = jnp.pad(array, padding)
    return array.reshape((n_chunks, chunk_size) + array.shape[1:])


def _paint_impl(
    mesh, positions, mass, resampler, scale, translate, period, gradient, chunk_size, xp
):
    shape = mesh.shape
    n = positions.shape[0]
    chunk_size = _get_chunk_size(chunk_size, n, resampler, len(shape))

    def paint_chunk(flat, positions_chunk, mass_chunk):
        index, weight = _stencil(
            positions_chunk,
            mass_chunk,
            shape,
            resampler,
            scale,
            translate,
            period,
            gradient,
            xp,
        )
        index, weight = index.ravel(), weight.ravel().astype(flat.dtype)
        if xp is np:
            painted = np.bincount(index, weights=weight, minlength=flat.size)
            return flat + painted.astype(flat.dtype)
        return flat.at[index].add(weight, mode="promise_in_bounds")

    flat = mesh.ravel()
    if xp is np:
        for start in range(0, n, chunk_size):
            chunk = slice(start, start + chunk_size)
            flat = paint_chunk(
                flat, positions[chunk], None if mass is None else mass[chunk]
            )
    elif n <= chunk_size:
        flat = paint_chunk(flat, positions, mass)
    else:
        if mass is None:
            mass = jnp.ones(n, dtype=positions.dtype)
        # Padded particles have zero mass and paint nothing.
        chunks = (_split_chunks(positions, chunk_size), _split_chunks(mass, chunk_size))
        flat, _ = jax.lax.scan(
            lambda flat, chunk: (paint_chunk(flat, *chunk), None), flat, chunks
        )
    return flat.reshape(shape)


def _readout_impl(
    mesh, positions, resampler, scale, translate, period, gradient, chunk_size, xp
):
    shape = mesh.shape
    n = positions.shape[0]
    chunk_size = _get_chunk_size(chunk_size, n, resampler, len(shape))
    flat = mesh.ravel()

    def readout_chunk(positions_chunk):
        index, weight = _stencil(
            positions_chunk,
            None,
            shape,
            resampler,
            scale,
            translate,
            period,
            gradient,
            xp,
        )
        return xp.sum(flat[index] * weight, axis=-1)

    if n <= chunk_size:
        return readout_chunk(positions)
    if xp is np:
        return np.concatenate(
            [
                readout_chunk(positions[s : s + chunk_size])
                for s in range(0, n, chunk_size)
            ]
        )
    values = jax.lax.map(readout_chunk, _split_chunks(positions, chunk_size))
    return values.reshape(-1)[:n]


def _paint_jax(
    mesh, positions, mass, resampler, scale, translate, period, gradient, chunk_size
):
    return _paint_impl(
        mesh,
        positions,
        mass,
        resampler,
        scale,
        translate,
        period,
        gradient,
        chunk_size,
        jnp,
    )


def _readout_jax(
    mesh, positions, resampler, scale, translate, period, gradient, chunk_size
):
    return _readout_impl(
        mesh, positions, resampler, scale, translate, period, gradient, chunk_size, jnp
    )


if jax_installed:
    _paint_jax = jax.jit(_paint_jax, static_argnames=_static_argnames)
    _readout_jax = jax.jit(_readout_jax, static_argnames=_static_argnames)


def _prepare(mesh, positions, mass, scale, translate, period, xp):
    """Cast inputs and broadcast per-axis parameters.

    Windows are evaluated in the precision of the positions (at least that
    of the mesh), as pmesh does in double. Only the accumulation uses the
    mesh precision.
    """
    ndim = mesh.ndim
    positions = xp.asarray(positions)
    dtype = xp.result_type(positions.dtype, mesh.dtype, float)
    positions = positions.astype(dtype).reshape(-1, ndim)
    if mass is not None:
        mass = xp.broadcast_to(xp.asarray(mass, dtype=dtype), positions.shape[:1])
    scale = xp.broadcast_to(xp.asarray(scale, dtype=dtype), (ndim,))
    translate = xp.broadcast_to(xp.asarray(translate, dtype=dtype), (ndim,))
    if period is None:
        period = mesh.shape
    period = tuple(int(p) for p in np.broadcast_to(period, (ndim,)))
    return positions, mass, scale, translate, period


def paint(
    mesh,
    positions,
    mass=None,
    resampler="cic",
    scale=1.0,
    translate=0.0,
    period=None,
    gradient=None,
    chunk_size=None,
    backend=None,
):
    """Paint particles onto a mesh, adding to its current values.

    Pure function equivalent to pmesh ``ResampleWindow.paint``. With the JAX
    backend it can be jitted, vmapped and differentiated with respect to
    ``mesh``, ``positions``, ``mass``, ``scale`` and ``translate``.

    Args:
        mesh (array-like): Mesh to add to, of any dimension ``ndim``. Its
            dtype sets the accumulation precision.
        positions (array-like): Positions with shape ``(N, ndim)``.
        mass (array-like, optional): Scalar or ``(N,)`` masses. Defaults to 1.
        resampler (str): ``nnb``, ``cic``, ``tsc`` or ``pcs`` (or an alias).
        scale (float or array-like): Factor converting positions to grid
            units, per axis. ``Nmesh / BoxSize`` for simulation units.
        translate (float or array-like): Shift in grid units, per axis.
        period (int or tuple[int], optional): Periodicity per axis in grid
            units, ``0`` for none. Defaults to the mesh shape. Nodes that fall
            outside the mesh after wrapping are dropped.
        gradient (int, optional): Axis along which to paint the derivative of
            the window (pmesh ``gradient``/``diffdir``).
        chunk_size (int, optional): Particles per chunk. Defaults to
            ``2**23 // support**ndim``.
        backend (str, optional): ``jax`` or ``numpy``, see
            :func:`get_array_module`.

    Returns:
        array: New mesh, ``mesh`` plus the painted particles.
    """
    xp = get_array_module(backend)
    mesh = xp.asarray(mesh)
    positions, mass, scale, translate, period = _prepare(
        mesh, positions, mass, scale, translate, period, xp
    )
    args = (get_resampler(resampler), scale, translate, period, gradient, chunk_size)
    if xp is np:
        return _paint_impl(mesh, positions, mass, *args, np)
    return _paint_jax(mesh, positions, mass, *args)


def readout(
    mesh,
    positions,
    resampler="cic",
    scale=1.0,
    translate=0.0,
    period=None,
    gradient=None,
    chunk_size=None,
    backend=None,
):
    """Interpolate a mesh at particle positions.

    Pure function equivalent to pmesh ``ResampleWindow.readout``, and the
    adjoint of :func:`paint`. With the JAX backend it can be jitted, vmapped
    and differentiated.

    Args:
        mesh (array-like): Mesh to interpolate, of any dimension ``ndim``.
        positions (array-like): Positions with shape ``(N, ndim)``.
        resampler (str): ``nnb``, ``cic``, ``tsc`` or ``pcs`` (or an alias).
        scale (float or array-like): Factor converting positions to grid units.
        translate (float or array-like): Shift in grid units, per axis.
        period (int or tuple[int], optional): Periodicity per axis in grid
            units, ``0`` for none. Defaults to the mesh shape.
        gradient (int, optional): Axis along which to read the derivative of
            the field interpolant.
        chunk_size (int, optional): Particles per chunk.
        backend (str, optional): ``jax`` or ``numpy``, see
            :func:`get_array_module`.

    Returns:
        array: Interpolated values with shape ``(N,)``.
    """
    xp = get_array_module(backend)
    mesh = xp.asarray(mesh)
    positions, _, scale, translate, period = _prepare(
        mesh, positions, None, scale, translate, period, xp
    )
    args = (get_resampler(resampler), scale, translate, period, gradient, chunk_size)
    if xp is np:
        return _readout_impl(mesh, positions, *args, np)
    return _readout_jax(mesh, positions, *args)


def _moments_impl(
    mesh,
    positions,
    groups,
    probability,
    object_weights,
    resampler,
    scale,
    translate,
    period,
    xp,
):
    shape = mesh.shape
    if positions.shape[0] == 0:
        return mesh, xp.zeros_like(mesh)

    index, window = _stencil(
        positions, None, shape, resampler, scale, translate, period, None, xp
    )
    stencil_size = index.shape[1]
    cell = index.ravel()
    group = xp.repeat(groups, stencil_size)
    first = xp.repeat(probability, stencil_size) * window.ravel()
    second = first * window.ravel()

    # Sort the (point, node) pairs by (object, node), so that each run of
    # equal pairs holds the contributions of one object to one node. A single
    # integer key sorted together with the values is the fastest option.
    flat_size = mesh.size
    num_objects = object_weights.shape[0]
    if xp is np or jax.config.jax_enable_x64:
        key_dtype = np.int64
    else:
        key_dtype = np.int32
    if num_objects * flat_size <= np.iinfo(key_dtype).max:
        key = group.astype(key_dtype) * flat_size + cell.astype(key_dtype)
        if xp is np:
            order = np.argsort(key, kind="stable")
            key, first, second = key[order], first[order], second[order]
        else:
            key, first, second = jax.lax.sort((key, first, second), num_keys=1)
        group, cell = key // flat_size, key % flat_size
    else:
        order = jnp.lexsort((cell, group))
        cell, group = cell[order], group[order]
        first, second = first[order], second[order]
    new_segment = xp.concatenate(
        [
            xp.ones(1, dtype=bool),
            (cell[1:] != cell[:-1]) | (group[1:] != group[:-1]),
        ]
    )
    segment = xp.cumsum(new_segment) - 1

    if xp is np:
        n_segments = int(segment[-1]) + 1
        first = np.bincount(segment, weights=first, minlength=n_segments)
        second = np.bincount(segment, weights=second, minlength=n_segments)
        segment_cell = cell[new_segment]
        segment_group = group[new_segment]
    else:
        # Static upper bound on the number of segments. Unused segments keep
        # zero moments and add nothing to node 0.
        n_segments = cell.shape[0]
        first = jax.ops.segment_sum(
            first, segment, num_segments=n_segments, indices_are_sorted=True
        )
        second = jax.ops.segment_sum(
            second, segment, num_segments=n_segments, indices_are_sorted=True
        )
        segment_cell = jnp.zeros(n_segments, cell.dtype).at[segment].set(cell)
        segment_group = jnp.zeros(n_segments, group.dtype).at[segment].set(group)

    weight = object_weights[segment_group]
    mean = weight * first
    # Cauchy-Schwarz guarantees second >= first**2; clip rounding errors.
    variance = weight**2 * xp.maximum(second - first * first, 0)

    if xp is np:
        mean = np.bincount(segment_cell, weights=mean, minlength=flat_size)
        variance = np.bincount(segment_cell, weights=variance, minlength=flat_size)
    else:
        zeros = jnp.zeros(flat_size, dtype=mean.dtype)
        mean = zeros.at[segment_cell].add(mean, mode="promise_in_bounds")
        variance = zeros.at[segment_cell].add(variance, mode="promise_in_bounds")
    return (
        mean.reshape(shape).astype(mesh.dtype),
        variance.reshape(shape).astype(mesh.dtype),
    )


def _moments_jax(
    mesh,
    positions,
    groups,
    probability,
    object_weights,
    resampler,
    scale,
    translate,
    period,
):
    return _moments_impl(
        mesh,
        positions,
        groups,
        probability,
        object_weights,
        resampler,
        scale,
        translate,
        period,
        jnp,
    )


if jax_installed:
    _moments_jax = jax.jit(_moments_jax, static_argnames=("resampler", "period"))


def paint_moments(
    shape,
    positions,
    groups,
    probability=None,
    object_weights=None,
    num_objects=None,
    resampler="cic",
    scale=1.0,
    translate=0.0,
    period=None,
    dtype="f8",
    backend=None,
):
    """Exact mean and variance meshes of a one-point-per-object random draw.

    Object ``i`` owns the points ``x_ik`` (rows of ``positions`` with
    ``groups == i``). A realisation draws one point per object, point ``k``
    with probability ``p_ik``, independently between objects, and paints it
    with mass ``w_i``. For every node ``c``, with ``W`` the window::

        a_ic = sum_k p_ik W(c, x_ik)
        b_ic = sum_k p_ik W(c, x_ik)**2
        mean_c = sum_i w_i a_ic
        variance_c = sum_i w_i**2 (b_ic - a_ic**2)

    This is the limit of the Monte Carlo mean and variance over infinitely
    many realisations, obtained with one sort of the ``(point, node)`` pairs.
    For ``nnb`` (``W**2 == W``), ``variance_c = sum_i w_i**2 a_ic (1 - a_ic)``,
    the Poisson binomial variance. Memory scales with the number of points
    times ``support**ndim``; there is no chunking.

    Args:
        shape (tuple[int]): Mesh shape.
        positions (array-like): Points of all objects, shape ``(P, ndim)``.
        groups (array-like): Object index of each point, integers in
            ``[0, num_objects)``, shape ``(P,)``. Need not be sorted.
        probability (array-like, optional): Draw probability ``p_ik`` of each
            point, shape ``(P,)``, summing to 1 within each object. Defaults
            to ``1 / n_i``, with ``n_i`` the number of points of object ``i``.
        object_weights (array-like, optional): Mass ``w_i`` of each object,
            shape ``(num_objects,)``. Defaults to 1.
        num_objects (int, optional): Number of objects. Defaults to
            ``max(groups) + 1``; required when ``groups`` is traced.
        resampler (str): ``nnb``, ``cic``, ``tsc`` or ``pcs`` (or an alias).
        scale (float or array-like): Factor converting positions to grid units.
        translate (float or array-like): Shift in grid units, per axis.
        period (int or tuple[int], optional): Periodicity per axis in grid
            units, ``0`` for none. Defaults to the mesh shape.
        dtype (str): Dtype of the returned meshes.
        backend (str, optional): ``jax`` or ``numpy``, see
            :func:`get_array_module`.

    Returns:
        tuple[array, array]: Mean and variance meshes.
    """
    xp = get_array_module(backend)
    mesh = xp.zeros(tuple(shape), dtype=dtype)
    positions, _, scale, translate, period = _prepare(
        mesh, positions, None, scale, translate, period, xp
    )
    groups = xp.asarray(groups).reshape(-1)
    if num_objects is None:
        num_objects = int(groups.max()) + 1 if groups.size else 0
    if probability is None:
        if xp is np:
            count = np.bincount(groups, minlength=num_objects)
        else:
            count = jnp.bincount(groups, length=num_objects)
        probability = 1.0 / count[groups]
    probability = xp.asarray(probability, dtype=positions.dtype).reshape(-1)
    if object_weights is None:
        object_weights = xp.ones(num_objects, dtype=positions.dtype)
    object_weights = xp.asarray(object_weights, dtype=positions.dtype).reshape(-1)

    args = (get_resampler(resampler), scale, translate, period)
    if xp is np:
        return _moments_impl(
            mesh, positions, groups, probability, object_weights, *args, np
        )
    return _moments_jax(mesh, positions, groups, probability, object_weights, *args)


def fwindow(w, resampler, backend=None):
    """Fourier transform of a 1D resampling window.

    Equal to ``sinc(w / 2) ** p`` with ``p`` the window order (1 for ``nnb``
    up to 4 for ``pcs``), see Jing (2005), eq. 18.

    Args:
        w (array-like): Circular frequency ``k * BoxSize / Nmesh``.
        resampler (str): Resampler name.
        backend (str, optional): ``jax`` or ``numpy``.

    Returns:
        array: Window values.
    """
    xp = get_array_module(backend)
    return xp.sinc(xp.asarray(w) / (2 * np.pi)) ** window_support(resampler)


def get_compensation(resampler, backend=None):
    """Return a pmesh-style function deconvolving the resampling window.

    The function has signature ``function(w, v)`` and is meant for
    ``ComplexField.apply(function, kind="circular")``.
    """

    def function(w, v):
        transfer = 1.0
        for wi in w:
            transfer = transfer * fwindow(wi, resampler, backend=backend)
        return v / transfer

    return function


class Affine(object):
    """Affine transformation from positions to grid units, as in pmesh.

    Args:
        ndim (int): Number of dimensions.
        scale (float or array-like): Factor multiplying positions.
        translate (float or array-like): Shift in grid units.
        period (int or array-like): Periodicity in grid units, ``0`` for none.
    """

    def __init__(self, ndim, scale=1.0, translate=0.0, period=0):
        self.ndim = ndim
        self.scale = np.broadcast_to(np.asarray(scale, dtype="f8"), (ndim,)).copy()
        self.translate = np.broadcast_to(
            np.asarray(translate, dtype="f8"), (ndim,)
        ).copy()
        self.period = np.broadcast_to(np.asarray(period, dtype="i8"), (ndim,)).copy()

    def rescale(self, amount):
        """Return a new Affine with the scale multiplied by ``amount``."""
        return Affine(self.ndim, self.scale * amount, self.translate, self.period)

    def shift(self, amount):
        """Return a new Affine with the translation shifted by ``amount``."""
        return Affine(self.ndim, self.scale, self.translate + amount, self.period)


class ParticleMesh(object):
    """Single-process particle mesh with the pmesh interface.

    Coordinates follow pmesh: ``x[d]`` holds the node positions
    ``i * BoxSize / Nmesh`` wrapped to ``[-BoxSize/2, BoxSize/2)``, and
    ``k[d]`` the wavenumbers of the ``rfftn`` layout (the last axis keeps
    ``Nmesh // 2 + 1`` modes). Each coordinate array is broadcastable to
    the field shape.

    Args:
        Nmesh (array-like): Number of mesh nodes per axis.
        BoxSize (float or array-like): Box size per axis.
        dtype (str): Real dtype of the mesh.
        resampler (str): Default resampler.
        backend (str, optional): ``jax`` or ``numpy``, see
            :func:`get_array_module`.
    """

    def __init__(self, Nmesh, BoxSize=1.0, dtype="f8", resampler="cic", backend=None):
        self.Nmesh = np.array(Nmesh, dtype="i8").reshape(-1)
        self.ndim = len(self.Nmesh)
        self.BoxSize = np.empty(self.ndim, dtype="f8")
        self.BoxSize[:] = BoxSize
        self.dtype = np.dtype(dtype)
        self.resampler = get_resampler(resampler)
        self.backend = backend
        # Transform from simulation unit to grid unit.
        self.affine = Affine(
            self.ndim, scale=1.0 * self.Nmesh / self.BoxSize, period=self.Nmesh
        )
        # Transform from grid unit to grid unit.
        self.affine_grid = Affine(self.ndim, scale=1.0, period=self.Nmesh)

        self.x, self.i = [], []
        self.k, self.w, self.o = [], [], []
        for d in range(self.ndim):
            n = self.Nmesh[d]
            axis_shape = np.ones(self.ndim, dtype="intp")
            axis_shape[d] = n
            ri = np.arange(n, dtype=self.dtype)
            ri[ri >= n // 2] -= n
            self.x.append((ri * self.BoxSize[d] / n).reshape(axis_shape))
            self.i.append(np.arange(n, dtype="intp").reshape(axis_shape))

            n_o = n // 2 + 1 if d == self.ndim - 1 else n
            axis_shape[d] = n_o
            wi = np.arange(n_o, dtype=self.dtype)
            wi[wi >= n // 2] -= n
            wi *= 2 * np.pi / n
            self.w.append(wi.reshape(axis_shape))
            self.k.append((wi * n / self.BoxSize[d]).reshape(axis_shape))
            self.o.append(np.arange(n_o, dtype="intp").reshape(axis_shape))

    @property
    def xp(self):
        return get_array_module(self.backend)

    @property
    def shape(self):
        return tuple(int(n) for n in self.Nmesh)

    @property
    def complex_shape(self):
        return self.shape[:-1] + (self.shape[-1] // 2 + 1,)

    def _key(self):
        return (
            self.shape,
            tuple(self.BoxSize.tolist()),
            self.dtype.str,
            self.resampler,
            self.backend,
        )

    def __eq__(self, other):
        return isinstance(other, ParticleMesh) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def create(self, type="real", value=None):
        """Create a field filled with ``value`` (zero by default).

        Args:
            type (str): ``real`` or ``complex``.
            value (array-like or scalar, optional): Initial values.

        Returns:
            RealField or ComplexField: New field.
        """
        xp = self.xp
        if type in ("real", RealField):
            field_class, shape, dtype = RealField, self.shape, self.dtype
        elif type in ("complex", ComplexField):
            field_class, shape = ComplexField, self.complex_shape
            dtype = np.result_type(self.dtype, np.complex64)
        else:
            raise ValueError(f"Unknown field type {type}")
        field_value = xp.zeros(shape, dtype=dtype)
        if value is not None:
            field_value = field_value + xp.asarray(value, dtype=dtype)
        return field_class(self, field_value)

    def paint(
        self,
        pos,
        mass=1.0,
        resampler=None,
        transform=None,
        hold=False,
        gradient=None,
        out=None,
        chunk_size=None,
    ):
        """Paint particles onto a real field.

        Args:
            pos (array-like): Positions with shape ``(N, ndim)`` in simulation
                units.
            mass (array-like): Scalar or ``(N,)`` masses.
            resampler (str, optional): Resampler, defaults to ``self.resampler``.
            transform (Affine, optional): Defaults to ``self.affine``.
            hold (bool): Add to the values of ``out`` instead of zeroing it.
            gradient (int, optional): Axis of the window derivative.
            out (RealField, optional): Field receiving the result. Its value is
                replaced, since JAX arrays are immutable.
            chunk_size (int, optional): Particles per chunk.

        Returns:
            RealField: Painted field (``out`` when given).
        """
        if transform is None:
            transform = self.affine
        if resampler is None:
            resampler = self.resampler
        if out is not None and hold:
            base = out.value
        else:
            base = self.xp.zeros(self.shape, dtype=self.dtype)
        value = paint(
            base,
            pos,
            mass=mass,
            resampler=resampler,
            scale=transform.scale,
            translate=transform.translate,
            period=tuple(transform.period),
            gradient=gradient,
            chunk_size=chunk_size,
            backend=self.backend,
        )
        if out is None:
            return RealField(self, value)
        out.value = value
        return out

    def paint_moments(
        self,
        pos,
        groups,
        probability=None,
        object_weights=None,
        num_objects=None,
        resampler=None,
        transform=None,
    ):
        """Exact mean and variance fields of a one-point-per-object draw.

        See :func:`paint_moments` for the arguments.

        Returns:
            tuple[RealField, RealField]: Mean and variance fields.
        """
        if transform is None:
            transform = self.affine
        if resampler is None:
            resampler = self.resampler
        mean, variance = paint_moments(
            self.shape,
            pos,
            groups,
            probability=probability,
            object_weights=object_weights,
            num_objects=num_objects,
            resampler=resampler,
            scale=transform.scale,
            translate=transform.translate,
            period=tuple(transform.period),
            dtype=self.dtype,
            backend=self.backend,
        )
        return RealField(self, mean), RealField(self, variance)


class _Field(object):
    """Mesh values attached to a :class:`ParticleMesh`."""

    def __init__(self, pm, value):
        self.pm = pm
        self.value = value

    @property
    def BoxSize(self):
        return self.pm.BoxSize

    @property
    def Nmesh(self):
        return self.pm.Nmesh

    @property
    def ndim(self):
        return self.pm.ndim

    @property
    def shape(self):
        return self.value.shape

    @property
    def dtype(self):
        return self.value.dtype

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.value, dtype=dtype)

    def __repr__(self):
        return f"{self.__class__.__name__}:{self.value!r}"

    def copy(self):
        return self.__class__(self.pm, self.value.copy())

    def tree_flatten(self):
        return (self.value,), self.pm

    @classmethod
    def tree_unflatten(cls, pm, children):
        return cls(pm, *children)


class RealField(_Field):
    """Real-space field. ``x`` holds the node coordinates and ``i`` the node
    indices, as in pmesh."""

    @property
    def x(self):
        return self.pm.x

    @property
    def i(self):
        return self.pm.i

    def paint(
        self, pos, mass=1.0, resampler=None, transform=None, hold=False, gradient=None
    ):
        """Paint particles onto this field, see :meth:`ParticleMesh.paint`."""
        return self.pm.paint(
            pos,
            mass=mass,
            resampler=resampler,
            transform=transform,
            hold=hold,
            gradient=gradient,
            out=self,
        )

    def readout(
        self, pos, resampler=None, transform=None, gradient=None, chunk_size=None
    ):
        """Interpolate the field at positions ``pos`` in simulation units."""
        if transform is None:
            transform = self.pm.affine
        if resampler is None:
            resampler = self.pm.resampler
        return readout(
            self.value,
            pos,
            resampler=resampler,
            scale=transform.scale,
            translate=transform.translate,
            period=tuple(transform.period),
            gradient=gradient,
            chunk_size=chunk_size,
            backend=self.pm.backend,
        )

    def r2c(self):
        """Forward FFT, normalised by ``1 / prod(Nmesh)`` as in pmesh.

        The complex field is in natural ``rfftn`` order.
        """
        norm = np.prod(self.pm.Nmesh**-1.0)
        return ComplexField(self.pm, self.pm.xp.fft.rfftn(self.value) * norm)

    def csum(self):
        return self.pm.xp.sum(self.value)

    def cmean(self):
        return self.pm.xp.mean(self.value)

    def apply(self, func, kind="relative"):
        """Return a new field with value ``func(coords, value)``.

        ``kind`` is ``relative`` for coordinates ``x`` or ``index`` for ``i``.
        """
        coords = {"relative": self.pm.x, "index": self.pm.i}[kind]
        return RealField(self.pm, func(coords, self.value))


class ComplexField(_Field):
    """Fourier-space field, in natural ``rfftn`` order. ``k`` holds the
    wavenumbers, ``w`` the circular frequencies and ``i`` the indices."""

    @property
    def k(self):
        return self.pm.k

    @property
    def w(self):
        return self.pm.w

    @property
    def i(self):
        return self.pm.o

    @property
    def real(self):
        return self.value.real

    @property
    def imag(self):
        return self.value.imag

    def c2r(self):
        """Inverse FFT, not normalised, so that ``c2r(r2c(f)) == f``."""
        norm = np.prod(self.pm.Nmesh * 1.0)
        value = self.pm.xp.fft.irfftn(self.value, s=self.pm.shape) * norm
        return RealField(self.pm, value.astype(self.pm.dtype))

    def apply(self, func, kind="wavenumber"):
        """Return a new field with value ``func(coords, value)``.

        ``kind`` is ``wavenumber`` for ``k``, ``circular`` for ``w`` or
        ``index`` for ``i``.
        """
        coords = {"wavenumber": self.pm.k, "circular": self.pm.w, "index": self.pm.o}
        return ComplexField(self.pm, func(coords[kind], self.value))


if jax_installed:
    jax.tree_util.register_pytree_node_class(RealField)
    jax.tree_util.register_pytree_node_class(ComplexField)


def _check_finite(values, name, xp):
    """Raise if ``values`` contain NaN/inf.

    The check is skipped when ``values`` are traced (under ``jit``, ``vmap``
    or ``lax.map``), since their content is unknown at trace time.
    """
    if values is None:
        return
    try:
        finite = bool(xp.all(xp.isfinite(xp.asarray(values))))
    except Exception as error:
        if jax_installed and isinstance(
            error,
            (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError),
        ):
            return
        raise
    if not finite:
        raise ValueError(f"Some {name} are NaN/inf")


def _static_geometry(value, name, ndim, dtype):
    """Convert a mesh geometry parameter to a numpy array of shape ``(ndim,)``.

    The geometry fixes the mesh shape, so it must be concrete even when the
    positions and weights are traced.
    """
    try:
        value = np.asarray(value, dtype=dtype)
    except Exception as error:
        if jax_installed and isinstance(
            error,
            (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError),
        ):
            raise ValueError(
                f"{name} must be concrete (not traced): it sets the mesh "
                "geometry. Compute it with numpy or outside jit/vmap/lax.map."
            ) from error
        raise
    return np.broadcast_to(value, (ndim,))


def _catalog_mesh(positions, boxsize, nmesh, boxcenter, resampler, dtype, backend):
    """Build the mesh of a catalog and shift positions to its corner.

    Positions stay on the array module of ``backend`` and may be traced; the
    geometry (``boxsize``, ``nmesh``, ``boxcenter``) must be concrete.

    Returns:
        tuple: ParticleMesh and positions minus
        ``offset = boxcenter - boxsize / 2``.
    """
    xp = get_array_module(backend)
    positions = xp.asarray(positions)
    ndim = positions.shape[-1]
    boxsize = _static_geometry(boxsize, "boxsize", ndim, "f8")
    boxcenter = _static_geometry(boxcenter, "boxcenter", ndim, "f8")
    nmesh = _static_geometry(nmesh, "nmesh", ndim, "i8")
    offset = boxcenter - boxsize / 2.0
    _check_finite(positions, "positions", xp)
    pm = ParticleMesh(
        Nmesh=nmesh,
        BoxSize=boxsize,
        dtype=dtype,
        resampler=resampler,
        backend=backend,
    )
    return pm, positions - xp.asarray(offset, dtype=positions.dtype)


def paint_catalog(
    positions,
    boxsize,
    nmesh,
    boxcenter=0.0,
    resampler="cic",
    weights=None,
    scaling=None,
    dtype="f8",
    chunk_size=None,
    backend=None,
):
    """Paint a weighted catalog onto a new mesh centred on ``boxcenter``.

    Replaces the pmesh sequence ``ParticleMesh(...)``, ``pm.create("real")``
    and ``pm.paint(positions - offset, ...)``, where
    ``offset = boxcenter - boxsize / 2``. Mesh node ``i`` therefore sits at
    ``offset + i * boxsize / nmesh``.

    Args:
        positions (array-like): Positions with shape ``(N, ndim)``.
        boxsize (float or array-like): Box size per axis.
        nmesh (int or array-like): Number of mesh nodes per axis.
        boxcenter (float or array-like): Box centre per axis.
        resampler (str): ``nnb``, ``cic``, ``tsc`` or ``pcs`` (or an alias).
        weights (array-like, optional): Per-object weights. Defaults to 1.
        scaling (float, optional): Extra factor multiplying the weights.
        dtype (str): Real dtype of the mesh.
        chunk_size (int, optional): Particles per chunk.
        backend (str, optional): ``jax`` or ``numpy``, see
            :func:`get_array_module`.

    Returns:
        RealField: Painted mesh.

    Raises:
        ValueError: If positions or weights contain non-finite values.
    """
    pm, positions = _catalog_mesh(
        positions, boxsize, nmesh, boxcenter, resampler, dtype, backend
    )
    if weights is None:
        weights = 1.0
    _check_finite(weights, "weights", pm.xp)
    if scaling is not None:
        weights = weights * scaling
    return pm.paint(positions, mass=weights, chunk_size=chunk_size)


def paint_catalog_moments(
    positions,
    groups,
    boxsize,
    nmesh,
    boxcenter=0.0,
    resampler="cic",
    probability=None,
    object_weights=None,
    num_objects=None,
    dtype="f8",
    backend=None,
):
    """Exact mean and variance meshes of a catalog of point distributions.

    Mesh geometry as in :func:`paint_catalog`, moments as in
    :func:`paint_moments`: each object ``i`` is represented by the points
    ``positions[groups == i]``, one of which is drawn per realisation.

    Args:
        positions (array-like): Points of all objects, shape ``(P, ndim)``.
        groups (array-like): Object index of each point, shape ``(P,)``.
        boxsize (float or array-like): Box size per axis.
        nmesh (int or array-like): Number of mesh nodes per axis.
        boxcenter (float or array-like): Box centre per axis.
        resampler (str): ``nnb``, ``cic``, ``tsc`` or ``pcs`` (or an alias).
        probability (array-like, optional): Draw probability of each point.
            Defaults to uniform within each object.
        object_weights (array-like, optional): Mass of each object.
            Defaults to 1.
        num_objects (int, optional): Number of objects.
        dtype (str): Real dtype of the meshes.
        backend (str, optional): ``jax`` or ``numpy``, see
            :func:`get_array_module`.

    Returns:
        tuple[RealField, RealField]: Mean and variance meshes.

    Raises:
        ValueError: If positions or weights contain non-finite values.
    """
    pm, positions = _catalog_mesh(
        positions, boxsize, nmesh, boxcenter, resampler, dtype, backend
    )
    for values in (probability, object_weights):
        _check_finite(values, "weights", pm.xp)
    return pm.paint_moments(
        positions,
        groups,
        probability=probability,
        object_weights=object_weights,
        num_objects=num_objects,
    )
