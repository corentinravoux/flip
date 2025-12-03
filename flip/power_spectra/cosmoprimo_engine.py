import numpy as np

from flip.utils import create_log

log = create_log()

try:
    from cosmoprimo import Fourier
except ImportError:
    log.add(
        "Install cosmoprimo https://github.com/cosmodesi/cosmoprimo to use cosmoprimo_engine.py module",
        level="warning",
    )


def get_fiducial_fs8(cosmology, redshift):
    return cosmology.sigma8_z(redshift) * cosmology.growth_rate(redshift)


def get_fiducial_s8(cosmology, redshift):
    return cosmology.sigma8_z(redshift)


def compute_power_spectrum(
    cosmology,
    redshift,
    minimal_wavenumber,
    maximal_wavenumber,
    number_points,
    non_linear_model=None,
    logspace=True,
):
    if type(cosmology) is dict:
        raise ValueError(
            "power_spectrum_settings should be an instance of Cosmology, not a dict"
        )
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

    fourier = Fourier(cosmology, engine="class")
    power_spectrum_linear = fourier.pk_interpolator(of=("delta_m"))(
        wavenumber, z=redshift
    )
    if non_linear_model is not None:
        power_spectrum_non_linear = fourier.pk_interpolator(
            of=("delta_m"), non_linear=True
        )(wavenumber, z=redshift)
    else:
        power_spectrum_non_linear = None

    fs8_fiducial = get_fiducial_fs8(cosmology, redshift)
    s8_fiducial = get_fiducial_s8(cosmology, redshift)
    fiducial = {"fsigma_8": fs8_fiducial, "sigma_8": s8_fiducial}

    if non_linear_model is None:
        power_spectrum_non_linear = None

    return (
        wavenumber,
        power_spectrum_linear,
        power_spectrum_non_linear,
        fiducial,
    )
