import numpy as np

from flip.utils import create_log

log = create_log()

try:
    from classy import Class
except ImportError:
    log.add(
        "Install CLASS https://github.com/lesgourg/class_public to use class_engine.py module",
        level="warning",
    )

_minimal_suggested_settings = ("h", "omega_b", "omega_cdm", "sigma8", "n_s")


_class_setting_default = {
    "output": "mPk",
    "N_ncdm": 0,
    "P_k_max_h/Mpc": 300,
}


_class_acuracy_setting_defaults = {
    "tau_reio": 0.0561,
    # "N_eff": 3.044, cause a bug with class since Nov. 2024
    "YHe": 0.24,
    "halofit_k_per_decade": 3000.0,
    "l_switch_limber": 40.0,
    "accurate_lensing": 1,
    "num_mu_minus_lmax": 1000.0,
    "delta_l_max": 1000.0,
    "tol_thermo_integration": 1.0e-5,
    "recfast_x_He0_trigger_delta": 0.01,
    "recfast_x_H0_trigger_delta": 0.01,
    "evolver": 0,
    "k_min_tau0": 0.002,
    "k_max_tau0_over_l_max": 10.0,
    "k_step_sub": 0.015,
    "k_step_super": 0.0001,
    "k_step_super_reduction": 0.1,
    "start_small_k_at_tau_c_over_tau_h": 0.0004,
    "start_large_k_at_tau_h_over_tau_k": 0.05,
    "tight_coupling_trigger_tau_c_over_tau_h": 0.005,
    "tight_coupling_trigger_tau_c_over_tau_k": 0.008,
    "start_sources_at_tau_c_over_tau_h": 0.006,
    "l_max_g": 50,
    "l_max_pol_g": 25,
    "l_max_ur": 50,
    "radiation_streaming_approximation": 2,
    "radiation_streaming_trigger_tau_over_tau_k": 240.0,
    "radiation_streaming_trigger_tau_c_over_tau": 100.0,
    "ur_fluid_approximation": 2,
    "ur_fluid_trigger_tau_over_tau_k": 50.0,
    "ncdm_fluid_approximation": 3,
    "ncdm_fluid_trigger_tau_over_tau_k": 51.0,
    "l_logstep": 1.026,
    "l_linstep": 25,
    "hyper_sampling_flat": 12.0,
    "hyper_sampling_curved_low_nu": 10.0,
    "hyper_sampling_curved_high_nu": 10.0,
    "hyper_nu_sampling_step": 10.0,
    "hyper_phi_min_abs": 1.0e-10,
    "hyper_x_tol": 1.0e-4,
    "hyper_flat_approximation_nu": 1.0e6,
    "q_linstep": 0.20,
    "q_logstep_spline": 20.0,
    "q_logstep_trapzd": 0.5,
    "q_numstep_transition": 250,
    "transfer_neglect_delta_k_S_t0": 100.0,
    "transfer_neglect_delta_k_S_t1": 100.0,
    "transfer_neglect_delta_k_S_t2": 100.0,
    "transfer_neglect_delta_k_S_e": 100.0,
    "transfer_neglect_delta_k_V_t1": 100.0,
    "transfer_neglect_delta_k_V_t2": 100.0,
    "transfer_neglect_delta_k_V_e": 100.0,
    "transfer_neglect_delta_k_V_b": 100.0,
    "transfer_neglect_delta_k_T_t2": 100.0,
    "transfer_neglect_delta_k_T_e": 100.0,
    "transfer_neglect_delta_k_T_b": 100.0,
    "neglect_CMB_sources_below_visibility": 1.0e-30,
    "transfer_neglect_late_source": 3000.0,
}


def get_fiducial_fs8(model, redshift):
    """Return fiducial $f\sigma_8$ from a CLASS model at redshift.

    Args:
        model (Class): Initialized and computed CLASS wrapper.
        redshift (float): Target redshift.

    Returns:
        float: Scale-independent $f\sigma_8(z)$.
    """
    return model.scale_independent_f_sigma8(redshift)


def get_fiducial_s8(model, redshift):
    """Return fiducial $\sigma_8$ from a CLASS model at redshift.

    Args:
        model (Class): Initialized and computed CLASS wrapper.
        redshift (float): Target redshift.

    Returns:
        float: $\sigma(R=8\,\mathrm{Mpc}/h, z)$.
    """
    return model.sigma(8 / model.h(), redshift)


def compute_power_spectrum(
    power_spectrum_settings,
    redshift,
    minimal_wavenumber,
    maximal_wavenumber,
    number_points,
    non_linear_model=None,
    logspace=True,
):
    """Compute linear/non-linear $P(k)$ using CLASS.

    Args:
        power_spectrum_settings (dict): CLASS settings (cosmology + outputs).
        redshift (float): Redshift for $P(k)$.
        minimal_wavenumber (float): Minimum $k$ in h/Mpc.
        maximal_wavenumber (float): Maximum $k$ in h/Mpc.
        number_points (int): Number of $k$ samples.
        non_linear_model (str|None): Enable non-linear engine in CLASS.
        logspace (bool): Sample $k$ in log-space when True.

    Returns:
        tuple: `(k, P_lin, P_nl_or_None, fiducial_dict)`.

    Raises:
        Exception: Propagates CLASS errors with a helpful message.
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

    if non_linear_model is not None:
        power_spectrum_settings.update(
            {
                "non linear": non_linear_model,
            }
        )
    power_spectrum_settings.update({"z_max_pk": redshift})

    class_settings = _class_acuracy_setting_defaults
    class_settings.update(_class_setting_default)
    class_settings.update(power_spectrum_settings)

    try:
        model = Class()
        model.set(class_settings)
        model.compute()

    except Exception as error:
        log.add(
            "The class computation of power spectrum did not work. "
            f"A minimal suggested setting is {_minimal_suggested_settings}"
        )
        raise error

    power_spectrum_linear, power_spectrum_non_linear = np.empty((2, number_points))

    for i, k in enumerate(wavenumber):
        power_spectrum_linear[i] = (
            model.pk_lin(k * model.h(), redshift) * model.h() ** 3
        )

        if non_linear_model is not None:
            power_spectrum_non_linear[i] = (
                model.pk(k * model.h(), redshift) * model.h() ** 3
            )

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
