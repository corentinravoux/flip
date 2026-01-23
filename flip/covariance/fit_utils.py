import os
import pickle

import numpy as np
import pandas as pd
import snsim
import snutils
from astropy.cosmology import FlatLambdaCDM

from flip.covariance import covariance, fitter
from flip.utils import create_log

log = create_log()


def fit_density_minuit(
    parameter_dict,
    model_name,
    likelihood_type,
    likelihood_properties,
    parameter_fit,
    overwrite=False,
    size_batch=10_000,
    number_worker=32,
    additional_parameters_values=(),
    maximum_number_coordinates=None,
):
    """Fit density field parameters with Minuit.

    Loads a grid from parquet, builds a density covariance via `flip.covariance`,
    constructs the likelihood, runs Minuit, and pickle-dumps fit results.

    Args:
        parameter_dict (dict): Parameter specs with values/errors/limits.
        model_name (str): Covariance model name.
        likelihood_type (str): Likelihood variant key.
        likelihood_properties (dict): Likelihood options.
        parameter_fit (tuple): `(grid_name, power_spectrum_dict, name_out_fit, str_fit)`.
        overwrite (bool): Overwrite existing pickle output.
        size_batch (int): Batch size for covariance generation.
        number_worker (int): Parallel workers for covariance generation.
        additional_parameters_values (tuple): Extra parameters for covariance.
        maximum_number_coordinates (int, optional): Early stop if grid too large.

    Returns:
        None: Writes results to `name_out_fit`.
    """
    grid_name = parameter_fit[0]
    power_spectrum_dict = parameter_fit[1]
    name_out_fit = parameter_fit[2]
    str_fit = parameter_fit[3]

    if (os.path.isfile(name_out_fit)) & (overwrite is False):
        log.add("Fit already performed")
        return

    grid = pd.read_parquet(grid_name)
    if (maximum_number_coordinates is not None) & (
        grid["ra"].size > maximum_number_coordinates
    ):
        log.add("Maximum number coordinate exceeded")
        return

    for i in range(len(str_fit)):
        log.add(str_fit[i])

    coordinates_density = np.array([grid["ra"], grid["dec"], grid["rcom_zobs"]])
    data_density = {
        "density": np.array(grid["density"]),
        "density_error": np.array(grid["density_error"]),
    }

    covariance_fit = covariance.CovMatrix.init_from_flip(
        model_name,
        "density",
        power_spectrum_dict,
        coordinates_density=coordinates_density,
        size_batch=size_batch,
        number_worker=number_worker,
        additional_parameters_values=additional_parameters_values,
    )

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        covariance_fit,
        data_density,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
    )

    minuit_fitter.run()

    params = [
        minuit_fitter.minuit.values.to_dict(),
        minuit_fitter.minuit.limits.to_dict(),
        minuit_fitter.minuit.errors.to_dict(),
        minuit_fitter.minuit.valid,
        minuit_fitter.minuit.accurate,
        minuit_fitter.minuit.fval,
    ]

    pickle.dump(params, open(name_out_fit, "wb"))


def fit_density_interp_sigg_minuit(
    parameter_dict,
    model_name,
    likelihood_type,
    likelihood_properties,
    interpolation_value_name,
    interpolation_value_range,
    parameter_fit,
    overwrite=False,
    size_batch=10_000,
    number_worker=32,
    maximum_number_coordinates=None,
):
    """Fit density with 1D interpolation over `sigg` using Minuit.

    Precomputes covariances over `interpolation_value_range`, interpolates during
    likelihood evaluation, and runs Minuit.

    Args:
        parameter_dict (dict): Parameter specs.
        model_name (str): Covariance model name.
        likelihood_type (str): Likelihood variant key.
        likelihood_properties (dict): Likelihood options.
        interpolation_value_name (str): Interpolation parameter name.
        interpolation_value_range (array-like): Grid of `sigg` values.
        parameter_fit (tuple): `(grid_name, power_spectrum_dict, name_out_fit, str_fit)`.
        overwrite (bool): Overwrite existing output.
        size_batch (int): Covariance generation batch size.
        number_worker (int): Parallel workers for covariance.
        maximum_number_coordinates (int, optional): Early stop if grid too large.

    Returns:
        None: Writes results to `name_out_fit`.
    """
    grid_name = parameter_fit[0]
    power_spectrum_dict = parameter_fit[1]
    name_out_fit = parameter_fit[2]
    str_fit = parameter_fit[3]

    if (os.path.isfile(name_out_fit)) & (overwrite is False):
        log.add("Fit already performed")
        return

    grid = pd.read_parquet(grid_name)
    if (maximum_number_coordinates is not None) & (
        grid["ra"].size > maximum_number_coordinates
    ):
        log.add("Maximum number coordinate exceeded")
        return

    for i in range(len(str_fit)):
        log.add(str_fit[i])

    coordinates_density = np.array([grid["ra"], grid["dec"], grid["rcom_zobs"]])
    data_density = {
        "density": np.array(grid["density"]),
        "density_error": np.array(grid["density_error"]),
    }

    covariance_list = []
    for sigg in interpolation_value_range:
        covariance_list.append(
            covariance.CovMatrix.init_from_flip(
                model_name,
                "density",
                power_spectrum_dict,
                coordinates_density=coordinates_density,
                size_batch=size_batch,
                number_worker=number_worker,
                additional_parameters_values=(sigg,),
            )
        )

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        covariance_list,
        data_density,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
        interpolation_value_name=interpolation_value_name,
        interpolation_value_range=interpolation_value_range,
    )

    minuit_fitter.run()

    params = [
        minuit_fitter.minuit.values.to_dict(),
        minuit_fitter.minuit.limits.to_dict(),
        minuit_fitter.minuit.errors.to_dict(),
        minuit_fitter.minuit.valid,
        minuit_fitter.minuit.accurate,
        minuit_fitter.minuit.fval,
    ]

    pickle.dump(params, open(name_out_fit, "wb"))


def fit_velocity_true_minuit(
    parameter_dict,
    likelihood_type,
    likelihood_properties,
    zmin,
    z_simu,
    photo_type,
    completeness_file,
    parameter_fit,
    overwrite=False,
    size_batch=10_000,
    number_worker=32,
):
    """Fit true peculiar velocities (no measurement error) with Minuit.

    Extracts light curves from simulation, applies selection masks, builds the
    velocity covariance with appropriate cosmological scaling, and runs Minuit.

    Args:
        parameter_dict (dict): Parameter specs.
        likelihood_type (str): Likelihood variant key.
        likelihood_properties (dict): Likelihood options.
        zmin (float): Minimum redshift for selection.
        z_simu (float): Simulation redshift for scaling.
        photo_type (str): Phototyping mode for selection.
        completeness_file (str): Path to completeness table.
        parameter_fit (tuple): `(sim_name, fit_name, zmax, power_spectrum_dict, name_out_fit, str_fit)`.
        overwrite (bool): Overwrite existing output.
        size_batch (int): Covariance generation batch size.
        number_worker (int): Parallel workers for covariance.

    Returns:
        None: Writes results to `name_out_fit`.
    """

    sim_name = parameter_fit[0]
    fit_name = parameter_fit[1]
    zmax = parameter_fit[2]
    power_spectrum_dict = parameter_fit[3]
    name_out_fit = parameter_fit[4]
    str_fit = parameter_fit[5]

    if (os.path.isfile(name_out_fit)) & (overwrite is False):
        log.add("Fit already performed")
        return
    for i in range(len(str_fit)):
        log.add(str_fit[i])

    fit = snsim.io_utils.open_fit(fit_name)
    sim = snsim.SimSample.fromFile(sim_name)
    _, sim_lcs = snutils.get_fitted_light_curves(sim, fit)

    mag_m, mag_M, completeness, _, _, _, _ = np.loadtxt(
        completeness_file,
        skiprows=1,
    ).T
    mag_x = (mag_m + mag_M) / 2

    detection_mask = snutils.return_detection_mask(sim_lcs)
    typing_mask = snutils.return_typing_mask(
        sim_lcs, mag_x, completeness, photo_type=photo_type
    )
    phase_mask = snutils.give_phasemask(sim_lcs, fit)
    light_curve_mask = snutils.give_mask(
        fit, phasemask=phase_mask, verbose=False, zrange=[zmin, zmax]
    )

    data_velocity = fit[typing_mask & detection_mask & light_curve_mask]

    cosmo = FlatLambdaCDM(
        H0=data_velocity.attrs["cosmo"]["H0"], Om0=data_velocity.attrs["cosmo"]["Om0"]
    )
    data_velocity["rcom_zobs"] = (
        cosmo.comoving_distance(data_velocity["zobs"]).value * cosmo.h
    )
    data_velocity["hubble_norm"] = cosmo.H(data_velocity["zobs"]) / cosmo.h

    samepos_list = snutils.find_samepos(data_velocity)
    same_pos_mask = np.ones(len(data_velocity), dtype="bool")
    if len(samepos_list) > 0:
        for same_pos in samepos_list:
            same_pos_mask[same_pos[1:]] = False

    data_velocity = data_velocity[same_pos_mask]

    coordinates_velocity = [
        data_velocity["ra"].values,
        data_velocity["dec"].values,
        data_velocity["rcom_zobs"].values,
    ]
    data_velocity_true = {
        "velocity": data_velocity["vpec"].values,
        "velocity_error": np.zeros(len(data_velocity["vpec"].values)),
    }

    covariance_scaling = float(((cosmo.H(z_simu) / (1 + z_simu)) / cosmo.H0) ** 2)

    cov = covariance.CovMatrix.init_from_flip(
        "carreres23",
        "velocity",
        power_spectrum_dict,
        coordinates_velocity=coordinates_velocity,
        size_batch=size_batch,
        number_worker=number_worker,
    )
    cov.covariance_dict["vv"] = covariance_scaling * cov.covariance_dict["vv"]

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        cov,
        data_velocity_true,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
    )

    minuit_fitter.run()

    params = [
        minuit_fitter.minuit.values.to_dict(),
        minuit_fitter.minuit.limits.to_dict(),
        minuit_fitter.minuit.errors.to_dict(),
        minuit_fitter.minuit.valid,
        minuit_fitter.minuit.accurate,
        minuit_fitter.minuit.fval,
        data_velocity.index.size,
    ]

    pickle.dump(params, open(name_out_fit, "wb"))


def fit_velocity_true_interp_sigu_minuit(
    parameter_dict,
    likelihood_type,
    likelihood_properties,
    interpolation_value_name,
    interpolation_value_range,
    zmin,
    z_simu,
    photo_type,
    completeness_file,
    parameter_fit,
    overwrite=False,
    size_batch=10_000,
    number_worker=32,
):
    """Fit true velocities with 1D interpolation over `sigu` using Minuit.

    Precomputes velocity covariances for multiple `sigu` values and interpolates
    during likelihood evaluation.

    Args:
        parameter_dict (dict): Parameter specs.
        likelihood_type (str): Likelihood variant key.
        likelihood_properties (dict): Likelihood options.
        interpolation_value_name (str): Name of interpolation parameter.
        interpolation_value_range (array-like): Grid of `sigu` values.
        zmin (float): Minimum redshift.
        z_simu (float): Simulation redshift for scaling.
        photo_type (str): Phototyping mode.
        completeness_file (str): Path to completeness table.
        parameter_fit (tuple): `(sim_name, fit_name, zmax, power_spectrum_dict_list, name_out_fit, str_fit)`.
        overwrite (bool): Overwrite existing output.
        size_batch (int): Covariance generation batch size.
        number_worker (int): Parallel workers for covariance.

    Returns:
        None
    """

    sim_name = parameter_fit[0]
    fit_name = parameter_fit[1]
    zmax = parameter_fit[2]
    power_spectrum_dict_list = parameter_fit[3]
    name_out_fit = parameter_fit[4]
    str_fit = parameter_fit[5]

    if (os.path.isfile(name_out_fit)) & (overwrite is False):
        log.add("Fit already performed")
        return
    for i in range(len(str_fit)):
        log.add(str_fit[i])

    fit = snsim.io_utils.open_fit(fit_name)
    sim = snsim.SimSample.fromFile(sim_name)
    _, sim_lcs = snutils.get_fitted_light_curves(sim, fit)

    mag_m, mag_M, completeness, _, _, _, _ = np.loadtxt(
        completeness_file,
        skiprows=1,
    ).T
    mag_x = (mag_m + mag_M) / 2

    detection_mask = snutils.return_detection_mask(sim_lcs)
    typing_mask = snutils.return_typing_mask(
        sim_lcs, mag_x, completeness, photo_type=photo_type
    )
    phase_mask = snutils.give_phasemask(sim_lcs, fit)
    light_curve_mask = snutils.give_mask(
        fit, phasemask=phase_mask, verbose=False, zrange=[zmin, zmax]
    )

    data_velocity = fit[typing_mask & detection_mask & light_curve_mask]

    cosmo = FlatLambdaCDM(
        H0=data_velocity.attrs["cosmo"]["H0"], Om0=data_velocity.attrs["cosmo"]["Om0"]
    )
    data_velocity["rcom_zobs"] = (
        cosmo.comoving_distance(data_velocity["zobs"]).value * cosmo.h
    )
    data_velocity["hubble_norm"] = cosmo.H(data_velocity["zobs"]) / cosmo.h

    samepos_list = snutils.find_samepos(data_velocity)
    same_pos_mask = np.ones(len(data_velocity), dtype="bool")
    if len(samepos_list) > 0:
        for same_pos in samepos_list:
            same_pos_mask[same_pos[1:]] = False

    data_velocity = data_velocity[same_pos_mask]

    coordinates_velocity = [
        data_velocity["ra"].values,
        data_velocity["dec"].values,
        data_velocity["rcom_zobs"].values,
    ]
    data_velocity_true = {
        "velocity": data_velocity["vpec"].values,
        "velocity_error": np.zeros(len(data_velocity["vpec"].values)),
    }

    covariance_fit_list = []

    covariance_scaling = float(((cosmo.H(z_simu) / (1 + z_simu)) / cosmo.H0) ** 2)
    for i in range(len(interpolation_value_range)):

        cov_sigu = covariance.CovMatrix.init_from_flip(
            "carreres23",
            "velocity",
            power_spectrum_dict_list[i],
            coordinates_velocity=coordinates_velocity,
            size_batch=size_batch,
            number_worker=number_worker,
        )
        cov_sigu.covariance_dict["vv"] = (
            covariance_scaling * cov_sigu.covariance_dict["vv"]
        )
        covariance_fit_list.append(cov_sigu)

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        covariance_fit_list,
        data_velocity_true,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
        interpolation_value_name=interpolation_value_name,
        interpolation_value_range=interpolation_value_range,
    )

    minuit_fitter.run()

    params = [
        minuit_fitter.minuit.values.to_dict(),
        minuit_fitter.minuit.limits.to_dict(),
        minuit_fitter.minuit.errors.to_dict(),
        minuit_fitter.minuit.valid,
        minuit_fitter.minuit.accurate,
        minuit_fitter.minuit.fval,
        data_velocity.index.size,
    ]

    pickle.dump(params, open(name_out_fit, "wb"))


def fit_velocity_estimated_minuit(
    parameter_dict,
    likelihood_type,
    likelihood_properties,
    zmin,
    z_simu,
    photo_type,
    completeness_file,
    parameter_fit,
    overwrite=False,
    size_batch=10_000,
    number_worker=32,
):
    """Fit estimated peculiar velocities (with measurement error) using Minuit.

    Similar to true velocity fit but uses observed velocity estimates with errors.

    Args:
        parameter_dict (dict): Parameter specs.
        likelihood_type (str): Likelihood variant key.
        likelihood_properties (dict): Likelihood options.
        zmin (float): Minimum redshift.
        z_simu (float): Simulation redshift for scaling.
        photo_type (str): Phototyping mode.
        completeness_file (str): Path to completeness table.
        parameter_fit (tuple): `(sim_name, fit_name, zmax, power_spectrum_dict, name_out_fit, str_fit)`.
        overwrite (bool): Overwrite existing output.
        size_batch (int): Covariance generation batch size.
        number_worker (int): Parallel workers for covariance.

        Returns:
            None
    """

    sim_name = parameter_fit[0]
    fit_name = parameter_fit[1]
    zmax = parameter_fit[2]
    power_spectrum_dict = parameter_fit[3]
    name_out_fit = parameter_fit[4]
    str_fit = parameter_fit[5]

    if (os.path.isfile(name_out_fit)) & (overwrite is False):
        log.add("Fit already performed")
        return
    for i in range(len(str_fit)):
        log.add(str_fit[i])

    fit = snsim.io_utils.open_fit(fit_name)
    sim = snsim.SimSample.fromFile(sim_name)
    _, sim_lcs = snutils.get_fitted_light_curves(sim, fit)

    mag_m, mag_M, completeness, _, _, _, _ = np.loadtxt(
        completeness_file,
        skiprows=1,
    ).T
    mag_x = (mag_m + mag_M) / 2

    detection_mask = snutils.return_detection_mask(sim_lcs)
    typing_mask = snutils.return_typing_mask(
        sim_lcs, mag_x, completeness, photo_type=photo_type
    )
    phase_mask = snutils.give_phasemask(sim_lcs, fit)
    light_curve_mask = snutils.give_mask(
        fit, phasemask=phase_mask, verbose=False, zrange=[zmin, zmax]
    )

    data_velocity = fit[typing_mask & detection_mask & light_curve_mask]

    cosmo = FlatLambdaCDM(
        H0=data_velocity.attrs["cosmo"]["H0"], Om0=data_velocity.attrs["cosmo"]["Om0"]
    )
    data_velocity["rcom_zobs"] = (
        cosmo.comoving_distance(data_velocity["zobs"]).value * cosmo.h
    )
    data_velocity["hubble_norm"] = cosmo.H(data_velocity["zobs"]) / cosmo.h

    samepos_list = snutils.find_samepos(data_velocity)
    same_pos_mask = np.ones(len(data_velocity), dtype="bool")
    if len(samepos_list) > 0:
        for same_pos in samepos_list:
            same_pos_mask[same_pos[1:]] = False

    data_velocity = data_velocity[same_pos_mask]

    coordinates_velocity = [
        data_velocity["ra"].values,
        data_velocity["dec"].values,
        data_velocity["rcom_zobs"].values,
    ]

    covariance_scaling = float(((cosmo.H(z_simu) / (1 + z_simu)) / cosmo.H0) ** 2)

    cov = covariance.CovMatrix.init_from_flip(
        "carreres23",
        "velocity",
        power_spectrum_dict,
        coordinates_velocity=coordinates_velocity,
        size_batch=size_batch,
        number_worker=number_worker,
    )
    cov.covariance_dict["vv"] = covariance_scaling * cov.covariance_dict["vv"]

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        cov,
        data_velocity,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
    )

    minuit_fitter.run()

    params = [
        minuit_fitter.minuit.values.to_dict(),
        minuit_fitter.minuit.limits.to_dict(),
        minuit_fitter.minuit.errors.to_dict(),
        minuit_fitter.minuit.valid,
        minuit_fitter.minuit.accurate,
        minuit_fitter.minuit.fval,
        data_velocity.index.size,
    ]

    pickle.dump(params, open(name_out_fit, "wb"))


def fit_velocity_estimated_interp_sigu_minuit(
    parameter_dict,
    likelihood_type,
    likelihood_properties,
    interpolation_value_name,
    interpolation_value_range,
    zmin,
    z_simu,
    photo_type,
    completeness_file,
    parameter_fit,
    overwrite=False,
    size_batch=10_000,
    number_worker=32,
):
    """Fit estimated velocities with 1D interpolation over `sigu` using Minuit.

    Args follow the non-interpolated version, with additional interpolation controls.

    Args:
        parameter_dict (dict): Parameter specs.
        likelihood_type (str): Likelihood variant key.
        likelihood_properties (dict): Likelihood options.
        interpolation_value_name (str): Interpolation parameter name.
        interpolation_value_range (array-like): Grid of `sigu` values.
        zmin (float): Minimum redshift.
        z_simu (float): Simulation redshift for scaling.
        photo_type (str): Phototyping mode.
        completeness_file (str): Path to completeness.
        parameter_fit (tuple): `(sim_name, fit_name, zmax, power_spectrum_dict_list, name_out_fit, str_fit)`.
        overwrite (bool): Overwrite existing output.
        size_batch (int): Covariance generation batch size.
        number_worker (int): Parallel workers.

    Returns:
        None
    """

    sim_name = parameter_fit[0]
    fit_name = parameter_fit[1]
    zmax = parameter_fit[2]
    power_spectrum_dict_list = parameter_fit[3]
    name_out_fit = parameter_fit[4]
    str_fit = parameter_fit[5]

    if (os.path.isfile(name_out_fit)) & (overwrite is False):
        log.add("Fit already performed")
        return
    for i in range(len(str_fit)):
        log.add(str_fit[i])

    fit = snsim.io_utils.open_fit(fit_name)
    sim = snsim.SimSample.fromFile(sim_name)
    _, sim_lcs = snutils.get_fitted_light_curves(sim, fit)

    mag_m, mag_M, completeness, _, _, _, _ = np.loadtxt(
        completeness_file,
        skiprows=1,
    ).T
    mag_x = (mag_m + mag_M) / 2

    detection_mask = snutils.return_detection_mask(sim_lcs)
    typing_mask = snutils.return_typing_mask(
        sim_lcs, mag_x, completeness, photo_type=photo_type
    )
    phase_mask = snutils.give_phasemask(sim_lcs, fit)
    light_curve_mask = snutils.give_mask(
        fit, phasemask=phase_mask, verbose=False, zrange=[zmin, zmax]
    )

    data_velocity = fit[typing_mask & detection_mask & light_curve_mask]

    cosmo = FlatLambdaCDM(
        H0=data_velocity.attrs["cosmo"]["H0"], Om0=data_velocity.attrs["cosmo"]["Om0"]
    )
    data_velocity["rcom_zobs"] = (
        cosmo.comoving_distance(data_velocity["zobs"]).value * cosmo.h
    )
    data_velocity["hubble_norm"] = cosmo.H(data_velocity["zobs"]) / cosmo.h

    samepos_list = snutils.find_samepos(data_velocity)
    same_pos_mask = np.ones(len(data_velocity), dtype="bool")
    if len(samepos_list) > 0:
        for same_pos in samepos_list:
            same_pos_mask[same_pos[1:]] = False

    data_velocity = data_velocity[same_pos_mask]

    coordinates_velocity = [
        data_velocity["ra"].values,
        data_velocity["dec"].values,
        data_velocity["rcom_zobs"].values,
    ]

    covariance_fit_list = []

    covariance_scaling = float(((cosmo.H(z_simu) / (1 + z_simu)) / cosmo.H0) ** 2)
    for i in range(len(interpolation_value_range)):

        cov_sigu = covariance.CovMatrix.init_from_flip(
            "carreres23",
            "velocity",
            power_spectrum_dict_list[i],
            coordinates_velocity=coordinates_velocity,
            size_batch=size_batch,
            number_worker=number_worker,
        )
        cov_sigu.covariance_dict["vv"] = (
            covariance_scaling * cov_sigu.covariance_dict["vv"]
        )
        covariance_fit_list.append(cov_sigu)

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        covariance_fit_list,
        data_velocity,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
        interpolation_value_name=interpolation_value_name,
        interpolation_value_range=interpolation_value_range,
    )

    minuit_fitter.run()

    params = [
        minuit_fitter.minuit.values.to_dict(),
        minuit_fitter.minuit.limits.to_dict(),
        minuit_fitter.minuit.errors.to_dict(),
        minuit_fitter.minuit.valid,
        minuit_fitter.minuit.accurate,
        minuit_fitter.minuit.fval,
        data_velocity.index.size,
    ]

    pickle.dump(params, open(name_out_fit, "wb"))


def fit_full_velocity_estimated_minuit(
    parameter_dict,
    model_name,
    likelihood_type,
    likelihood_properties,
    zmin,
    z_simu,
    photo_type,
    completeness_file,
    parameter_fit,
    overwrite=False,
    size_batch=10_000,
    number_worker=32,
    additional_parameters_values=(),
    maximum_number_coordinates=None,
):
    """Joint density-velocity fit with cross-terms using Minuit.

    Builds full covariance blocks (gg, gv, vv), applies cosmological scaling
    to velocity terms, and runs Minuit.

    Args:
        parameter_dict (dict): Parameter specs.
        model_name (str): Covariance model name.
        likelihood_type (str): Likelihood variant key.
        likelihood_properties (dict): Likelihood options.
        zmin (float): Minimum redshift for SN selection.
        z_simu (float): Simulation redshift for scaling.
        photo_type (str): Phototyping mode.
        completeness_file (str): Path to completeness table.
        parameter_fit (tuple): `(sim_name, fit_name, grid_name, zmax, power_spectrum_dict, name_out_fit, str_fit)`.
        overwrite (bool): Overwrite existing output.
        size_batch (int): Covariance generation batch size.
        number_worker (int): Parallel workers for covariance.
        additional_parameters_values (tuple): Extra parameters for covariance.
        maximum_number_coordinates (int, optional): Early stop for large grids.

    Returns:
        None
    """

    sim_name = parameter_fit[0]
    fit_name = parameter_fit[1]
    grid_name = parameter_fit[2]
    zmax = parameter_fit[3]
    power_spectrum_dict = parameter_fit[4]
    name_out_fit = parameter_fit[5]
    str_fit = parameter_fit[6]

    if (os.path.isfile(name_out_fit)) & (overwrite is False):
        log.add("Fit already performed")
        return

    grid = pd.read_parquet(grid_name)
    if (maximum_number_coordinates is not None) & (
        grid["ra"].size > maximum_number_coordinates
    ):
        log.add("Maximum number coordinate exceeded")
        return

    for i in range(len(str_fit)):
        log.add(str_fit[i])

    coordinates_density = np.array([grid["ra"], grid["dec"], grid["rcom_zobs"]])
    data_density = {
        "density": np.array(grid["density"]),
        "density_error": np.array(grid["density_error"]),
    }

    fit = snsim.io_utils.open_fit(fit_name)
    sim = snsim.SimSample.fromFile(sim_name)
    _, sim_lcs = snutils.get_fitted_light_curves(sim, fit)

    mag_m, mag_M, completeness, _, _, _, _ = np.loadtxt(
        completeness_file,
        skiprows=1,
    ).T
    mag_x = (mag_m + mag_M) / 2

    detection_mask = snutils.return_detection_mask(sim_lcs)
    typing_mask = snutils.return_typing_mask(
        sim_lcs, mag_x, completeness, photo_type=photo_type
    )
    phase_mask = snutils.give_phasemask(sim_lcs, fit)
    light_curve_mask = snutils.give_mask(
        fit, phasemask=phase_mask, verbose=False, zrange=[zmin, zmax]
    )

    data_velocity = fit[typing_mask & detection_mask & light_curve_mask]

    cosmo = FlatLambdaCDM(
        H0=data_velocity.attrs["cosmo"]["H0"], Om0=data_velocity.attrs["cosmo"]["Om0"]
    )
    data_velocity["rcom_zobs"] = (
        cosmo.comoving_distance(data_velocity["zobs"]).value * cosmo.h
    )
    data_velocity["hubble_norm"] = cosmo.H(data_velocity["zobs"]) / cosmo.h

    samepos_list = snutils.find_samepos(data_velocity)
    same_pos_mask = np.ones(len(data_velocity), dtype="bool")
    if len(samepos_list) > 0:
        for same_pos in samepos_list:
            same_pos_mask[same_pos[1:]] = False

    data_velocity = data_velocity[same_pos_mask]

    data_full = {}
    data_full.update(data_density)
    for key in data_velocity.columns:
        data_full[key] = np.array(data_velocity[key])

    coordinates_velocity = [
        data_velocity["ra"].values,
        data_velocity["dec"].values,
        data_velocity["rcom_zobs"].values,
    ]

    covariance_scaling = float(((cosmo.H(z_simu) / (1 + z_simu)) / cosmo.H0))

    cov = covariance.CovMatrix.init_from_flip(
        model_name,
        "full",
        power_spectrum_dict,
        coordinates_density=coordinates_density,
        coordinates_velocity=coordinates_velocity,
        size_batch=size_batch,
        number_worker=number_worker,
        additional_parameters_values=additional_parameters_values,
    )

    cov.covariance_dict["vv"] = covariance_scaling**2 * cov.covariance_dict["vv"]
    cov.covariance_dict["gv"] = covariance_scaling * cov.covariance_dict["gv"]

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        cov,
        data_full,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
    )

    minuit_fitter.run()

    params = [
        minuit_fitter.minuit.values.to_dict(),
        minuit_fitter.minuit.limits.to_dict(),
        minuit_fitter.minuit.errors.to_dict(),
        minuit_fitter.minuit.valid,
        minuit_fitter.minuit.accurate,
        minuit_fitter.minuit.fval,
        data_velocity.index.size,
    ]

    pickle.dump(params, open(name_out_fit, "wb"))


def fit_full_velocity_estimated_interp_sigu_minuit(
    parameter_dict,
    model_name,
    likelihood_type,
    likelihood_properties,
    interpolation_value_name,
    interpolation_value_range,
    zmin,
    z_simu,
    photo_type,
    completeness_file,
    parameter_fit,
    overwrite=False,
    size_batch=10_000,
    number_worker=32,
    additional_parameters_values=(),
    maximum_number_coordinates=None,
):
    """Joint density-velocity fit with 1D interpolation over `sigu` using Minuit.

    Args mirror the non-interpolated full fit plus interpolation controls.

    Args:
        parameter_dict (dict): Parameter specs.
        model_name (str): Covariance model name.
        likelihood_type (str): Likelihood variant key.
        likelihood_properties (dict): Likelihood options.
        interpolation_value_name (str): Interpolation parameter name.
        interpolation_value_range (array-like): Grid of values.
        zmin (float): Minimum redshift.
        z_simu (float): Simulation redshift for scaling.
        photo_type (str): Phototyping mode.
        completeness_file (str): Path to completeness.
        parameter_fit (tuple): `(sim_name, fit_name, grid_name, zmax, power_spectrum_dict_list, name_out_fit, str_fit)`.
        overwrite (bool): Overwrite existing output.
        size_batch (int): Covariance generation batch size.
        number_worker (int): Parallel workers.
        additional_parameters_values (tuple): Extra parameters for covariance.
        maximum_number_coordinates (int, optional): Early stop.

    Returns:
        None
    """

    sim_name = parameter_fit[0]
    fit_name = parameter_fit[1]
    grid_name = parameter_fit[2]
    zmax = parameter_fit[3]
    power_spectrum_dict_list = parameter_fit[4]
    name_out_fit = parameter_fit[5]
    str_fit = parameter_fit[6]

    if (os.path.isfile(name_out_fit)) & (overwrite is False):
        log.add("Fit already performed")
        return

    grid = pd.read_parquet(grid_name)
    if (maximum_number_coordinates is not None) & (
        grid["ra"].size > maximum_number_coordinates
    ):
        log.add("Maximum number coordinate exceeded")
        return

    for i in range(len(str_fit)):
        log.add(str_fit[i])

    coordinates_density = np.array([grid["ra"], grid["dec"], grid["rcom_zobs"]])
    data_density = {
        "density": np.array(grid["density"]),
        "density_error": np.array(grid["density_error"]),
    }

    fit = snsim.io_utils.open_fit(fit_name)
    sim = snsim.SimSample.fromFile(sim_name)
    _, sim_lcs = snutils.get_fitted_light_curves(sim, fit)

    mag_m, mag_M, completeness, _, _, _, _ = np.loadtxt(
        completeness_file,
        skiprows=1,
    ).T
    mag_x = (mag_m + mag_M) / 2

    detection_mask = snutils.return_detection_mask(sim_lcs)
    typing_mask = snutils.return_typing_mask(
        sim_lcs, mag_x, completeness, photo_type=photo_type
    )
    phase_mask = snutils.give_phasemask(sim_lcs, fit)
    light_curve_mask = snutils.give_mask(
        fit, phasemask=phase_mask, verbose=False, zrange=[zmin, zmax]
    )

    data_velocity = fit[typing_mask & detection_mask & light_curve_mask]

    cosmo = FlatLambdaCDM(
        H0=data_velocity.attrs["cosmo"]["H0"], Om0=data_velocity.attrs["cosmo"]["Om0"]
    )
    data_velocity["rcom_zobs"] = (
        cosmo.comoving_distance(data_velocity["zobs"]).value * cosmo.h
    )
    data_velocity["hubble_norm"] = cosmo.H(data_velocity["zobs"]) / cosmo.h

    samepos_list = snutils.find_samepos(data_velocity)
    same_pos_mask = np.ones(len(data_velocity), dtype="bool")
    if len(samepos_list) > 0:
        for same_pos in samepos_list:
            same_pos_mask[same_pos[1:]] = False

    data_velocity = data_velocity[same_pos_mask]

    data_full = {}
    data_full.update(data_density)
    for key in data_velocity.columns:
        data_full[key] = np.array(data_velocity[key])

    coordinates_velocity = [
        data_velocity["ra"].values,
        data_velocity["dec"].values,
        data_velocity["rcom_zobs"].values,
    ]

    covariance_fit_list = []

    covariance_scaling = float(((cosmo.H(z_simu) / (1 + z_simu)) / cosmo.H0))

    for i in range(len(interpolation_value_range)):
        cov_sigu = covariance.CovMatrix.init_from_flip(
            model_name,
            "full",
            power_spectrum_dict_list[i],
            coordinates_density=coordinates_density,
            coordinates_velocity=coordinates_velocity,
            size_batch=size_batch,
            number_worker=number_worker,
            additional_parameters_values=additional_parameters_values,
        )

        cov_sigu.covariance_dict["vv"] = (
            covariance_scaling**2 * cov_sigu.covariance_dict["vv"]
        )
        cov_sigu.covariance_dict["gv"] = (
            covariance_scaling * cov_sigu.covariance_dict["gv"]
        )
        covariance_fit_list.append(cov_sigu)

    minuit_fitter = fitter.FitMinuit.init_from_covariance(
        covariance_fit_list,
        data_full,
        parameter_dict,
        likelihood_type=likelihood_type,
        likelihood_properties=likelihood_properties,
        interpolation_value_name=interpolation_value_name,
        interpolation_value_range=interpolation_value_range,
    )
    minuit_fitter.run()

    params = [
        minuit_fitter.minuit.values.to_dict(),
        minuit_fitter.minuit.limits.to_dict(),
        minuit_fitter.minuit.errors.to_dict(),
        minuit_fitter.minuit.valid,
        minuit_fitter.minuit.accurate,
        minuit_fitter.minuit.fval,
        data_velocity.index.size,
    ]

    pickle.dump(params, open(name_out_fit, "wb"))
