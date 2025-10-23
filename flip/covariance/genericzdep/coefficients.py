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
    coefficients_dict["vv"] = []
    return coefficients_dict


def get_diagonal_coefficients(parameter_values_dict, model_kind):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict


def get_cov_matrix_prefactor(D_growth_z):
    Delta_D_growth_squared = jnp.subtract.outer(D_growth_z, D_growth_z) ** 2
    matrix_cov_prefactor = {"vv": [1, Delta_D_growth_squared, Delta_D_growth_squared**2]}
    return matrix_cov_prefactor
