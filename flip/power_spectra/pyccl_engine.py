import numpy as np

from flip.utils import create_log

log = create_log()

try:
    import pyccl as ccl
except ImportError:
    pass


_minimal_suggested_settings = ("h", "Omega_b", "Omega_c", "sigma8", "n_s")

_pyccl_setting_default = {}


def get_fiducial_s8(model, redshift):
    r"""Return fiducial $\sigma_8$ using PyCCL at redshift.

    Args:
        model (ccl.Cosmology): PyCCL cosmology.
        redshift (float): Target redshift.

    Returns:
        float: $\sigma_8(z)$.
    """
    return model.sigmaR(8 / model.to_dict()["h"], 1 / (1 + redshift))


def get_fiducial_fs8(model, redshift):
    r"""Return fiducial $f\sigma_8$ using PyCCL at redshift.

    Args:
        model (ccl.Cosmology): PyCCL cosmology.
        redshift (float): Target redshift.

    Returns:
        float: $f(z)\,\sigma_8(z)$ (using PyCCL conventions).
    """
    return model.growth_rate(1 / (1 + redshift)) * get_fiducial_s8(model, redshift)


def compute_power_spectrum(
    power_spectrum_settings,
    redshift,
    minimal_wavenumber,
    maximal_wavenumber,
    number_points,
    non_linear_model=None,
    logspace=True,
):
    """Compute linear/non-linear $P(k)$ using PyCCL.

    Args:
        power_spectrum_settings (dict|ccl.Cosmology): PyCCL settings (cosmology
            + flags), or an already built cosmology (e.g. the one prepared by
            firecrown, possibly a ``CosmologyCalculator``).
        redshift (float): Redshift for $P(k)$.
        minimal_wavenumber (float): Minimum $k$ in h/Mpc.
        maximal_wavenumber (float): Maximum $k$ in h/Mpc.
        number_points (int): Number of $k$ samples.
        non_linear_model (str|None): PyCCL matter power spectrum model key. For
            a ``ccl.Cosmology`` input, any non-None value uses the non-linear
            model of that cosmology.
        logspace (bool): Sample $k$ in log-space when True.

    Returns:
        tuple: `(k, P_lin, P_nl_or_None, fiducial_dict)`.

    Raises:
        Exception: Propagates PyCCL errors with a helpful message.
    """
    if logspace:
        wavenumber = np.logspace(
            np.log10(minimal_wavenumber),
            np.log10(maximal_wavenumber),
            number_points,
        )
    else:
        wavenumber = np.linspace(
            minimal_wavenumber,
            maximal_wavenumber,
            number_points,
        )

    try:
        if isinstance(power_spectrum_settings, ccl.Cosmology):
            model = power_spectrum_settings
        else:
            pyccl_settings = {**_pyccl_setting_default, **power_spectrum_settings}
            if non_linear_model is not None:
                pyccl_settings["matter_power_spectrum"] = non_linear_model
            model = ccl.Cosmology(**pyccl_settings)

        h = model["h"]
        scale_factor = 1 / (1 + redshift)
        power_spectrum_linear = (
            ccl.linear_matter_power(model, wavenumber * h, scale_factor) * h**3
        )
        power_spectrum_non_linear = None
        if non_linear_model is not None:
            power_spectrum_non_linear = (
                ccl.nonlin_matter_power(model, wavenumber * h, scale_factor) * h**3
            )

    except Exception as error:
        log.add(
            "The pyccl computation of power spectrum did not work. "
            f"A minimal suggested setting is {_minimal_suggested_settings}"
        )
        raise error

    fs8_fiducial = get_fiducial_fs8(model, redshift)
    s8_fiducial = get_fiducial_s8(model, redshift)
    fiducial = {"fsigma_8": fs8_fiducial, "sigma_8": s8_fiducial}

    return (
        wavenumber,
        power_spectrum_linear,
        power_spectrum_non_linear,
        fiducial,
    )
