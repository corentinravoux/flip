import importlib
import os

import numpy as np

_available_engines = ["class_engine", "cosmoprimo_engine", "pyccl_engine"]
_available_power_spectrum_model = ["linearbel", "nonlinearbel", "linear"]
_available_power_spectrum_normalizaton = [
    "no_normalization",
    "growth_rate",
    "growth_amplitude",
]


def get_power_spectrum_suffix(
    redshift,
    minimal_wavenumber,
    maximal_wavenumber,
    number_points,
    log_space,
):
    """Build a filename suffix encoding spectrum sampling settings.

    Returns:
        str: Suffix including z, kmin, kmax, N, and log/lin tag.
    """
    return f"z{redshift}_kmin{minimal_wavenumber:.4f}_kmax{maximal_wavenumber:.4f}_N{number_points}{'_log' if log_space else '_lin'}"


def get_power_spectrum_name(
    power_spectrum_model,
    power_spectrum_type,
    suffix,
):
    """Construct a power spectrum filename for saving.

    Args:
        power_spectrum_model (str): Model name, e.g., `linearbel`.
        power_spectrum_type (str): One of `mm`, `mt`, `tt`.
        suffix (str): Sampling suffix from `get_power_spectrum_suffix`.

    Returns:
        str: Filename `power_spectrum_<model>_<type>_<suffix>.txt`.
    """
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
    """Save a power spectrum to disk as a two-row text file.

    Args:
        wavenumber (ndarray): $k$ samples.
        power_spectrum (ndarray): Spectrum values.
        power_spectrum_model (str): Model name.
        power_spectrum_type (str): `mm`, `mt`, or `tt`.
        suffix (str): Sampling suffix.
        header (str): Header string with metadata.
        path (str): Directory where to save.
    """
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
    normalization_power_spectrum="no_normalization",
    power_spectrum_non_linear_model=None,
    power_spectrum_model="linearbel",
    save_path=None,
):
    """Compute and optionally save MM/MT/TT power spectra.

    Args:
        power_spectrum_engine (str): Engine module, one of `_available_engines`.
        power_spectrum_settings (dict|object): Engine configuration.
        redshift (float): Target redshift.
        minimal_wavenumber (float): Minimum $k$ in h/Mpc.
        maximal_wavenumber (float): Maximum $k$ in h/Mpc.
        number_points (int): Number of $k$ samples.
        logspace (bool): Sample $k$ in log-space if True.
        normalization_power_spectrum (str): One of `_available_power_spectrum_normalizaton`.
        power_spectrum_non_linear_model (str|None): Non-linear engine flag for CLASS.
        power_spectrum_model (str): One of `_available_power_spectrum_model`.
        save_path (str|None): Directory to save spectra.

    Returns:
        tuple: `(k, P_mm, P_mt, P_tt, fiducial_dict)`.

    Raises:
        ValueError: If engine or model name is unsupported.
    """
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

    engine = importlib.import_module(f"flip.power_spectra.{power_spectrum_engine}")

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
    module = importlib.import_module("flip.power_spectra.models")
    model_function = getattr(module, f"get_{power_spectrum_model}_model")

    power_spectrum_mm, power_spectrum_mt, power_spectrum_tt = model_function(
        wavenumber,
        power_spectrum_linear,
        power_spectrum_non_linear=power_spectrum_non_linear,
        **fiducial,
    )

    if normalization_power_spectrum == "growth_rate":
        power_spectrum_mm = power_spectrum_mm
        power_spectrum_mt = power_spectrum_mt * (
            fiducial["fsigma_8"] / fiducial["sigma_8"]
        )
        power_spectrum_tt = (
            power_spectrum_tt * (fiducial["fsigma_8"] / fiducial["sigma_8"]) ** 2
        )
    elif normalization_power_spectrum == "growth_amplitude":
        power_spectrum_mm = power_spectrum_mm / fiducial["sigma_8"] ** 2
        power_spectrum_mt = power_spectrum_mt / fiducial["sigma_8"] ** 2
        power_spectrum_tt = power_spectrum_tt / fiducial["sigma_8"] ** 2
    elif normalization_power_spectrum == "no_normalization":
        pass
    else:
        raise ValueError(
            f"The normalization {normalization_power_spectrum} of the power spectrum is not available,"
            f"Please choose in {_available_power_spectrum_normalizaton}."
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

    return wavenumber, power_spectrum_mm, power_spectrum_mt, power_spectrum_tt, fiducial
