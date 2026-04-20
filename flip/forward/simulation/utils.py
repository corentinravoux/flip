import jax.numpy as jnp


def radial_velocity_from_velocity_position(
    velocities,
    positions,
    centroid=(0, 0, 0),
):
    # create the norm unit
    di, dj, dk = jnp.asarray(positions) - jnp.asarray(centroid)[:, None]
    normed = jnp.sqrt(jnp.sum(jnp.stack([di**2, dj**2, dk**2]), axis=0))

    # direction units
    direction_unit = jnp.stack([di, dj, dk]) / normed

    vradial = jnp.sum(velocities * direction_unit, axis=0)
    return vradial
