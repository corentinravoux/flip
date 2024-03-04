import os

import numpy as np

from flip.power_spectra import class_engine, models

_available_engines = ["class_engine"]
_available_power_spectrum_model = ["linearbel", "nonlinearbel"]


def get_power_spectrum_suffix(
    redshift,
    minimal_wavenumber,
    maximal_wavenumber,
    number_points,
    log_space,
):
    return f"z{redshift}_kmin{minimal_wavenumber:.4f}_kmax{maximal_wavenumber:.4f}_N{number_points}{'_log' if log_space else '_lin'}"


def get_power_spectrum_name(
    power_spectrum_model,
    power_spectrum_type,
    suffix,
):
    return f"power_spectrum_{power_spectrum_model}_{power_spectrum_type}_{suffix}.txt"


def save_power_spectrum(
    wavenumber,
    power_spectrum,
    power_spectrum_model,
    power_spectrum_type,
    suffix,
    header,
    path,
):
    power_spectrum_name = get_power_spectrum_name(
        power_spectrum_model,
        power_spectrum_type,
        suffix,
    )

    np.savetxt(
        os.path.join(path, power_spectrum_name),
        [wavenumber, power_spectrum],
        header=header,
    )


def compute_power_spectra(
    power_spectrum_engine,
    power_spectrum_settings,
    redshift,
    minimal_wavenumber,
    maximal_wavenumber,
    number_points,
    logspace=True,
    normalize_power_spectrum=True,
    power_spectrum_non_linear_model=None,
    power_spectrum_model="linearbel",
    save_path=None,
):
    if power_spectrum_engine not in _available_engines:
        raise ValueError(
            f"The engine {power_spectrum_engine} is not available"
            f"PLease choose between {_available_engines}"
        )
    if power_spectrum_model not in _available_power_spectrum_model:
        raise ValueError(
            f"The power spectrum model {power_spectrum_model} is not available"
            f"PLease choose between {_available_power_spectrum_model}"
        )

    engine = eval(power_spectrum_engine)

    (
        wavenumber,
        power_spectrum_linear,
        power_spectrum_non_linear,
        fiducial,
    ) = engine.compute_power_spectrum(
        power_spectrum_settings,
        redshift,
        minimal_wavenumber,
        maximal_wavenumber,
        number_points,
        logspace=logspace,
        non_linear_model=power_spectrum_non_linear_model,
    )

    if normalize_power_spectrum:
        power_spectrum_linear = power_spectrum_linear / fiducial["sigma_8"] ** 2
        if power_spectrum_non_linear is not None:
            power_spectrum_non_linear = (
                power_spectrum_non_linear / fiducial["sigma_8"] ** 2
            )

    power_spectrum_mm, power_spectrum_mt, power_spectrum_tt = eval(
        f"models.get_{power_spectrum_model}_model"
    )(
        wavenumber,
        power_spectrum_linear,
        power_spectrum_non_linear=power_spectrum_non_linear,
        **fiducial,
    )

    if save_path is not None:
        suffix = get_power_spectrum_suffix(
            redshift,
            minimal_wavenumber,
            maximal_wavenumber,
            number_points,
            logspace,
        )
        header = ""
        for key in fiducial.keys():
            header = header + f"fiducial {key} = {fiducial[key]} & "

        save_power_spectrum(
            wavenumber,
            power_spectrum_mm,
            power_spectrum_model,
            "mm",
            suffix,
            header,
            save_path,
        )
        save_power_spectrum(
            wavenumber,
            power_spectrum_mt,
            power_spectrum_model,
            "mt",
            suffix,
            header,
            save_path,
        )
        save_power_spectrum(
            wavenumber,
            power_spectrum_tt,
            power_spectrum_model,
            "tt",
            suffix,
            header,
            save_path,
        )

    return wavenumber, power_spectrum_mm, power_spectrum_mt, power_spectrum_tt
