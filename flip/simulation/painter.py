"""Extract observables from density and velocity fields.

Given the ``(density_field, velocity_field)`` output of
:meth:`~flip.simulation.generator.ForwardModel.get_fields`, this module provides:

* **CIC interpolation** at arbitrary positions — a custom implementation
  compatible with arbitrary batch sizes and JAX tracing.
* **Line-of-sight projection** ``v_los = v · r̂``, with safe zero-norm handling.
* **Sky ↔ Cartesian** coordinate conversions (RA/Dec in radians, flip convention).
* **RSD positions**: shift galaxies along LOS by ``v_los / (aH)`` for
  redshift-space mock catalogs.

Coordinate conventions (consistent with flip throughout):
  * RA / Dec in **radians**
  * Comoving distance in Mpc/h
  * Velocities in km/s
  * Positions passed to interpolation routines in Mpc/h (Cartesian, box frame)
"""

try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp


def _cic_read(grid_mesh, positions):
    """CIC-interpolate a 3D scalar field at arbitrary positions.

    Custom JAX-differentiable trilinear interpolation compatible with
    arbitrary batch sizes (unlike some jaxpm versions whose ``cic_read``
    requires positions to match the grid shape).  Periodic BCs applied.

    Args:
        grid_mesh (jnp.ndarray): Scalar field of shape ``(Nx, Ny, Nz)``.
        positions (jnp.ndarray): Positions in mesh-cell units ``[0, Ni)``,
            shape ``(N, 3)``.

    Returns:
        jnp.ndarray: Interpolated values at each position, shape ``(N,)``.
    """
    pos = jnp.expand_dims(positions, -2)  # (N, 1, 3)

    offsets = jnp.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
        dtype=jnp.float32,
    )[jnp.newaxis]  # (1, 8, 3)

    floor_pos = jnp.floor(pos)                # (N, 1, 3)
    neighbours = floor_pos + offsets          # (N, 8, 3)
    kernel = 1.0 - jnp.abs(pos - neighbours)  # (N, 8, 3)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]  # (N, 8)

    grid_shape = jnp.array(grid_mesh.shape)
    idx = jnp.mod(neighbours.astype(jnp.int32), grid_shape)  # (N, 8, 3)
    values = grid_mesh[idx[..., 0], idx[..., 1], idx[..., 2]]  # (N, 8)
    return (values * kernel).sum(axis=-1)  # (N,)


def _safe_normalize(positions):
    """Return unit vectors along each position vector; zero vector → zero.

    Args:
        positions (jnp.ndarray): Cartesian positions [Mpc/h], shape ``(N, 3)``.

    Returns:
        jnp.ndarray: Unit vectors, shape ``(N, 3)``.
    """
    r = jnp.linalg.norm(positions, axis=-1, keepdims=True)
    return positions / jnp.where(r > 0, r, jnp.ones_like(r))


def compute_los_velocity(velocities, positions):
    """Project 3D velocities onto the line-of-sight direction.

    Observer at the Cartesian origin; LOS direction is r̂ = pos / |pos|.

    Args:
        velocities (jnp.ndarray): Peculiar velocities [km/s], shape ``(N, 3)``.
        positions (jnp.ndarray): Galaxy positions [Mpc/h], shape ``(N, 3)``.

    Returns:
        jnp.ndarray: Line-of-sight velocities [km/s], shape ``(N,)``.
    """
    los_unit = _safe_normalize(positions)
    return jnp.sum(velocities * los_unit, axis=-1)


def interpolate_velocity_to_positions(velocity_field, positions, box_size, mesh_shape):
    """CIC-interpolate a 3D velocity field at galaxy positions.

    Args:
        velocity_field (jnp.ndarray): Velocity field [km/s], shape
            ``(*mesh_shape, 3)``.
        positions (jnp.ndarray): Galaxy positions [Mpc/h], shape ``(N, 3)``.
            Coordinates must lie within ``[0, box_size)`` along each axis.
        box_size (array-like): Box dimensions [Mpc/h], shape ``(3,)``.
        mesh_shape (array-like): Grid resolution, shape ``(3,)``.

    Returns:
        jnp.ndarray: Velocity at each galaxy [km/s], shape ``(N, 3)``.
    """
    box_size = jnp.array(box_size, dtype=jnp.float32)
    mesh_arr = jnp.array(mesh_shape, dtype=jnp.float32)
    pos_mesh = positions / box_size * mesh_arr  # Mpc/h → mesh units
    return jnp.stack(
        [_cic_read(velocity_field[..., i], pos_mesh) for i in range(3)],
        axis=-1,
    )


def interpolate_density_to_positions(density_field, positions, box_size, mesh_shape):
    """CIC-interpolate the density contrast δ(x) at galaxy positions.

    Args:
        density_field (jnp.ndarray): Density contrast δ, shape ``mesh_shape``.
        positions (jnp.ndarray): Galaxy positions [Mpc/h], shape ``(N, 3)``.
        box_size (array-like): Box dimensions [Mpc/h].
        mesh_shape (array-like): Grid resolution.

    Returns:
        jnp.ndarray: δ at each galaxy, shape ``(N,)``.
    """
    box_size = jnp.array(box_size, dtype=jnp.float32)
    mesh_arr = jnp.array(mesh_shape, dtype=jnp.float32)
    pos_mesh = positions / box_size * mesh_arr
    return _cic_read(density_field, pos_mesh)


def sky_to_cartesian(ra, dec, rcom):
    """Convert sky coordinates to Cartesian positions in Mpc/h.

    RA/Dec in **radians** (flip convention).

    Args:
        ra (jnp.ndarray): Right ascension in radians, shape ``(N,)``.
        dec (jnp.ndarray): Declination in radians, shape ``(N,)``.
        rcom (jnp.ndarray): Comoving distance in Mpc/h, shape ``(N,)``.

    Returns:
        jnp.ndarray: Cartesian positions [Mpc/h], shape ``(N, 3)``.
    """
    cos_dec = jnp.cos(dec)
    x = rcom * cos_dec * jnp.cos(ra)
    y = rcom * cos_dec * jnp.sin(ra)
    z = rcom * jnp.sin(dec)
    return jnp.stack([x, y, z], axis=-1)


def cartesian_to_box_frame(positions, box_size, center_offset=None):
    """Shift Cartesian positions from observer-centred to box-corner frame.

    The simulation box has its corner at the origin; the observer sits at the
    box centre (or at ``center_offset`` if provided).

    Args:
        positions (jnp.ndarray): Observer-centred Cartesian positions [Mpc/h],
            shape ``(N, 3)``.
        box_size (array-like): Box dimensions [Mpc/h].
        center_offset (jnp.ndarray | None): Position of the observer relative
            to the box corner [Mpc/h]. Defaults to the box centre (L/2, L/2, L/2).

    Returns:
        jnp.ndarray: Box-frame positions [Mpc/h], shape ``(N, 3)``.
    """
    box = jnp.array(box_size, dtype=jnp.float32)
    if center_offset is None:
        center_offset = box / 2.0
    return positions + center_offset


def apply_rsd(positions_box, velocities, cosmo, a, box_size, los_axis=2):
    """Shift positions along the LOS by the plane-parallel RSD displacement.

    Applies  x_rsd = x + v_los / (a H(a))  in Mpc/h, then wraps periodically.

    Args:
        positions_box (jnp.ndarray): Galaxy positions in box frame [Mpc/h],
            shape ``(N, 3)``.
        velocities (jnp.ndarray): Peculiar velocities [km/s], shape ``(N, 3)``
            or ``(N,)`` if already projected on ``los_axis``.
        cosmo (jax_cosmo.Cosmology): Cosmological parameters.
        a (float): Scale factor.
        box_size (array-like): Box dimensions [Mpc/h].
        los_axis (int): Axis along which to apply RSD (0, 1, or 2). Default 2.

    Returns:
        jnp.ndarray: Redshift-space positions [Mpc/h], shape ``(N, 3)``.
    """
    from jax_cosmo import background

    a_arr = jnp.atleast_1d(a)
    E = jnp.sqrt(background.Esqr(cosmo, a_arr))[0]
    H_a = 100.0 * cosmo.h * E  # km/s / (Mpc/h)

    v_los = velocities if velocities.ndim == 1 else velocities[:, los_axis]
    rsd_shift = v_los / (a * H_a)  # Mpc/h

    box = jnp.array(box_size, dtype=jnp.float32)
    new_pos = positions_box.at[:, los_axis].add(rsd_shift)
    return new_pos % box
