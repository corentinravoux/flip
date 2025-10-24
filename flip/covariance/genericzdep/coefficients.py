from ...config import __use_jax__

if __use_jax__:
    try:
        import jax.numpy as jnp
        from jax import jit

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
    redshift_dict=None,
):
    coefficients_dict = {}
    coefficients_dict["vv"] = [
        1.0,
        1.0,
        1.0,
    ]
    return coefficients_dict


def get_diagonal_coefficients(parameter_values_dict, model_kind):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict


def get_cov_matrix_prefactor(z, f_z, E_z, D_growth_z):
    Ai_z = f_z * D_growth_z * E_z / (1 + z)
    Aij_z = jnp.outer(Ai_z, Ai_z)
    Delta_D_growth_squared = jnp.subtract.outer(D_growth_z, D_growth_z) ** 2

    matrix_cov_prefactor = {
        "vv": [
            Aij_z * jnp.ones(Delta_D_growth_squared.shape),
            Aij_z * Delta_D_growth_squared,
            Aij_z * 1 / 2 * Delta_D_growth_squared**2,
        ]
    }
    return matrix_cov_prefactor
