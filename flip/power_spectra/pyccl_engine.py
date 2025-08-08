import numpy as np

from flip.utils import create_log

log = create_log()

try:
    import pyccl as ccl
except:
    log.add(
        "Install CCL https://github.com/LSSTDESC/CCL to use pyccl_engine.py module",
        level="warning",
    )


_minimal_suggested_settings = ("h", "Omega_b", "Omega_c", "sigma8", "n_s")

_pyccl_setting_default = {}


def get_fiducial_fs8(model, redshift):
    return model.growth_rate(1 / (1 + redshift))


def get_fiducial_s8(model, redshift):
    return model.sigmaR(8 / model.to_dict()["h"], 1 / (1 + redshift))


def compute_power_spectrum(
    power_spectrum_settings,
    redshift,
    minimal_wavenumber,
    maximal_wavenumber,
    number_points,
    non_linear_model=None,
    logspace=True,
):
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

    if non_linear_model is not None:
        power_spectrum_settings.update(
            {
                "matter_power_spectrum": non_linear_model,
            }
        )

    pyccl_settings = _pyccl_setting_default
    pyccl_settings.update(power_spectrum_settings)

    try:
        model = ccl.Cosmology(**pyccl_settings)

        power_spectrum_linear = (
            ccl.linear_matter_power(
                model, wavenumber * model.to_dict()["h"], 1 / (1 + redshift)
            )
            * model.to_dict()["h"] ** 3
        )
        power_spectrum_non_linear = (
            ccl.nonlin_matter_power(
                model, wavenumber * model.to_dict()["h"], 1 / (1 + redshift)
            )
            * model.to_dict()["h"] ** 3
        )

    except Exception as error:
        log.add(
            "The class computation of power spectrum did not work. "
            f"A minimal suggested setting is {_minimal_suggested_settings}"
        )
        raise error

    fs8_fiducial = get_fiducial_fs8(model, redshift)
    s8_fiducial = get_fiducial_s8(model, redshift)
    fiducial = {"fsigma_8": fs8_fiducial, "sigma_8": s8_fiducial}

    if non_linear_model is None:
        power_spectrum_non_linear = None

    return (
        wavenumber,
        power_spectrum_linear,
        power_spectrum_non_linear,
        fiducial,
    )
