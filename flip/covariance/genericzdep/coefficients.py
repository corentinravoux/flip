from ..._config import __use_jax__

if __use_jax__:
    try:
        import jax.numpy as jnp

        jax_installed = True

    except ImportError:
        import numpy as jnp

        jax_installed = False
else:
    import numpy as jnp

    jax_installed = False


def get_coefficients(
    parameter_values_dict,
    model_kind,
    variant=None,
    covariance_prefactor_dict=None,
):
    coefficients_dict = {}

    redshift_velocity = covariance_prefactor_dict["redshift_velocity"]
    f_z = covariance_prefactor_dict["f_z"]
    E_z = covariance_prefactor_dict["E_z"]
    D_growth_z = covariance_prefactor_dict["D_growth_z"]

    coefficients_dict["vv"] = get_cov_matrix_prefactor(
        redshift_velocity, f_z, E_z, D_growth_z
    )
    return coefficients_dict


def get_cov_matrix_prefactor(redshift, f_z, E_z, D_growth_z):
    Ai_z = f_z * D_growth_z * E_z / (1 + redshift)
    Aij_z = jnp.outer(Ai_z, Ai_z)
    Delta_D_growth_squared = jnp.subtract.outer(D_growth_z, D_growth_z) ** 2

    return [
        Aij_z * jnp.ones_like(Delta_D_growth_squared),
        Aij_z * Delta_D_growth_squared,
        Aij_z * 1 / 2 * Delta_D_growth_squared**2,
    ]


def get_diagonal_coefficients(parameter_values_dict, model_kind):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict
