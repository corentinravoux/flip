from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


@partial(jax.jit, static_argnames=("box_size", "number_bins", "order"))
def field_from_grid(field, positions, box_size, number_bins, order=1):
    """CIC-interpolate a regular-grid field at arbitrary positions.

    field      : (N, N, N) scalar  e.g. density
              or (N, N, N, C) vector e.g. velocity (C=3).
    positions  : (M, 3) physical coords (Mpc/h), box centred on origin.
    order      : 0 = NGP, 1 = CIC (trilinear). Periodic box (mode='wrap').
    returns    : (M,) for scalar, (M, C) for vector.
    """
    dx = box_size / number_bins
    coords = ((positions + box_size / 2.0) / dx).T  # (3, M) grid-index coords
    coords = [coords[0], coords[1], coords[2]]

    def interp(comp):
        return map_coordinates(comp, coords, order=order, mode="wrap")

    if field.ndim == 3:  # scalar field
        return interp(field)  # (M,)
    return jax.vmap(interp, in_axes=-1, out_axes=-1)(field)  # (M, C) vector field


@partial(jax.jit, static_argnames=("box_size", "number_bins", "order"))
def radial_velocity_from_grid(
    velocity,
    comoving_distance_targets,
    line_of_sight,
    box_size,
    number_bins,
    order=1,
):
    """Interpolate velocity field at targets, project onto line of sight -> (M,)."""
    positions = comoving_distance_targets[:, None] * line_of_sight
    v = field_from_grid(velocity, positions, box_size, number_bins, order)
    return jnp.sum(v * line_of_sight, axis=-1)


@partial(jax.jit, static_argnames=("box_size", "number_bins", "order"))
def density_from_grid(
    density,
    comoving_distance_targets,
    line_of_sight,
    box_size,
    number_bins,
    order=1,
):
    """Interpolate density field at target positions (distance * line of sight) -> (M,)."""
    positions = comoving_distance_targets[:, None] * line_of_sight
    return field_from_grid(density, positions, box_size, number_bins, order)
