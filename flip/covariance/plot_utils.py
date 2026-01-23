import glob
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from flip import utils
from flip.covariance import cov_utils


def plot_1d_contraction(
    contraction,
    parameter_dict,
    rs_multiplied=True,
):
    """Plot 1D slices of contraction sums for gg/gv/vv blocks.

    Args:
        contraction (object): Contraction object with `compute_contraction_sum` and `coordinates_dict`.
        parameter_dict (dict): Parameter values for evaluation.
        rs_multiplied (bool): Multiply gg by r^2 for visualization.
    """
    contraction_sum = contraction.compute_contraction_sum(parameter_dict)
    coord = contraction.coordinates_dict

    index_min_perpendicular = np.argmin(np.abs(coord["r_perpendicular"][:, 0]))
    index_min_parallel = np.argmin(np.abs(coord["r_parallel"][0, :]))

    _, ax = plt.subplots(1, 3, figsize=(17, 5))

    if contraction.model_kind in ["density", "density_velocity", "full"]:
        xi_gg = contraction_sum["gg"]
        ax_plot = ax[0]

        if rs_multiplied:
            ax_plot.plot(
                coord["r_perpendicular"][:, index_min_parallel],
                (coord["r"] ** 2 * xi_gg)[:, index_min_parallel],
            )
            ax_plot.plot(
                coord["r_parallel"][index_min_perpendicular, :],
                (coord["r"] ** 2 * xi_gg)[index_min_perpendicular, :],
            )
            ax_plot.set_ylabel(r"$r_{i}^2 C_{gg}(r_{i})$", fontsize=15)
        else:
            ax_plot.plot(
                coord["r_perpendicular"][:, index_min_parallel],
                xi_gg[:, index_min_parallel],
            )
            ax_plot.plot(
                coord["r_parallel"][index_min_perpendicular, :],
                xi_gg[index_min_perpendicular, :],
            )
            ax_plot.set_ylabel(r"$C_{gg}(r_{i})$", fontsize=15)

        ax_plot.set_ylabel(r"$r_{i}^2 C_{gg}(r_{i})$", fontsize=15)
        ax_plot.set_xlabel(r"$r_{i}$", fontsize=15)
        ax_plot.legend([r"$\parallel$", r"$\bot$"], fontsize=15)

    if contraction.model_kind == "full":
        xi_gv = contraction_sum["gv"]
        ax_plot = ax[1]
        ax_plot.plot(
            coord["r_perpendicular"][:, index_min_parallel],
            xi_gv[:, index_min_parallel],
        )
        ax_plot.plot(
            coord["r_parallel"][index_min_perpendicular, :],
            xi_gv[index_min_perpendicular, :],
        )
        ax_plot.set_ylabel(r"$C_{gv}(r_{i})$", fontsize=15)
        ax_plot.set_xlabel(r"$r_{i}$", fontsize=15)
        ax_plot.legend([r"$\parallel$", r"$\bot$"], fontsize=15)

    if contraction.model_kind in ["velocity", "density_velocity", "full"]:
        xi_vv = contraction_sum["vv"]
        ax_plot = ax[2]
        ax_plot.plot(
            coord["r_perpendicular"][:, index_min_parallel],
            xi_vv[:, index_min_parallel],
        )
        ax_plot.plot(
            coord["r_parallel"][index_min_perpendicular, :],
            xi_vv[index_min_perpendicular, :],
        )
        ax_plot.set_ylabel(r"$C_{vv}(r_{i})$", fontsize=15)
        ax_plot.set_xlabel(r"$r_{i}$", fontsize=15)
        ax_plot.legend([r"$\parallel$", r"$\bot$"], fontsize=15)


def plot_2d_contraction(
    contraction,
    parameter_dict,
    rs_multiplied=True,
):
    """Plot 2D images of contraction sums for gg/gv/vv blocks.

    Args:
        contraction (object): Contraction with `compute_contraction_sum` & `coordinates_dict`.
        parameter_dict (dict): Parameter values for evaluation.
        rs_multiplied (bool): Multiply gg by r^2 for visualization.
    """
    contraction_sum = contraction.compute_contraction_sum(parameter_dict)
    coord = contraction.coordinates_dict

    r_perpendicular_min = np.min(coord["r_perpendicular"])
    r_perpendicular_max = np.max(coord["r_perpendicular"])
    r_parallel_min = np.min(coord["r_parallel"])
    r_parallel_max = np.max(coord["r_parallel"])
    extent = [
        r_perpendicular_min,
        r_perpendicular_max,
        r_parallel_max,
        r_parallel_min,
    ]

    _, ax = plt.subplots(1, 3, figsize=(17, 5))

    if contraction.model_kind in ["density", "density_velocity", "full"]:
        xi_gg = contraction_sum["gg"]

        ax_plot = ax[0]
        if rs_multiplied:
            image = ax_plot.imshow(
                np.transpose(coord["r"] ** 2 * xi_gg),
                extent=extent,
            )
            ax_plot.set_title(r"$r^2 C_{gg}(r)$", fontsize=15)
        else:
            image = ax_plot.imshow(
                np.transpose(xi_gg),
                extent=extent,
            )
            ax_plot.set_title(r"$C_{gg}(r)$", fontsize=15)

        plt.colorbar(image, ax=ax_plot)
        ax_plot.set_xlabel(r"$r_{\bot}$")
        ax_plot.set_ylabel(r"$r_{\parallel}$")
        ax_plot.set_title(r"$r^2 C_{gg}(r)$", fontsize=15)

    if contraction.model_kind == "full":
        xi_gv = contraction_sum["gv"]
        ax_plot = ax[1]

        image = ax_plot.imshow(
            np.transpose(xi_gv),
            extent=extent,
        )
        plt.colorbar(image, ax=ax_plot)
        ax_plot.set_xlabel(r"$r_{\bot}$")
        ax_plot.set_ylabel(r"$r_{\parallel}$")
        ax_plot.set_title(r"$C_{gv}(r)$", fontsize=15)

    if contraction.model_kind in ["velocity", "density_velocity", "full"]:
        xi_vv = contraction_sum["vv"]
        ax_plot = ax[2]

        image = ax_plot.imshow(
            np.transpose(xi_vv),
            extent=extent,
        )
        plt.colorbar(image, ax=ax_plot)
        ax_plot.set_xlabel(r"$r_{\bot}$")
        ax_plot.set_ylabel(r"$r_{\parallel}$")
        ax_plot.set_title(r"$C_{vv}(r)$", fontsize=15)


def plot_correlation_from_likelihood(
    likelihood,
    parameter_dict,
    covariance_prefactor_dict=None,
    **kwargs,
):
    """Plot correlation matrix computed from a likelihood’s covariance.

    Args:
        likelihood (BaseLikelihood): Likelihood instance providing data/covariance.
        parameter_dict (dict): Parameter specs; values read to form vector and variance.
        covariance_prefactor_dict (dict, optional): Prefactors per covariance block.
        **kwargs: Plot options, e.g., `vmin`, `vmax`.
    """
    vmin = utils.return_key(kwargs, "vmin", -0.1)
    vmax = utils.return_key(kwargs, "vmax", 0.1)

    parameter_names = likelihood.parameter_names
    parameter_values = [
        parameter_dict[parameters]["value"] for parameters in parameter_names
    ]
    parameter_values_dict = dict(zip(parameter_names, parameter_values))

    _, vector_variance = likelihood.data.give_data_and_variance(
        parameter_values_dict,
    )
    covariance_sum = likelihood.covariance.compute_covariance_sum(
        parameter_values_dict,
        vector_variance,
        covariance_prefactor_dict=covariance_prefactor_dict,
    )

    correlation_sum = cov_utils.return_correlation_matrix(covariance_sum)
    plt.imshow(correlation_sum, vmin=vmin, vmax=vmax)
    plt.colorbar()


def plot_all_fits(
    fit_output,
    parameters,
    fiducials=None,
    compute_fs8_from_beta=False,
    subset_plot=None,
    remove_lower=None,
    remove_higher=None,
    plot=True,
    **kwargs,
):
    """Scatter all fit results per parameter with errors.

    Args:
        fit_output (str): Directory containing pickled fit outputs.
        parameters (list[str]): Parameters to plot.
        fiducials (list[float]|None): Optional reference lines per parameter.
        compute_fs8_from_beta (bool): Plot `fs8 = beta_f * bs8` when requested.
        subset_plot (list[str]|None): Only include fit files whose names contain these substrings.
        remove_lower (dict|None): Exclude fits where param < threshold.
        remove_higher (dict|None): Exclude fits where param > threshold.
        plot (bool): Whether to render the plots.
        **kwargs: Matplotlib options, e.g., `figsize`.

    Returns:
        tuple: `(fit_name_to_plot, param_dict, error_dict)`.
    """

    fit_to_plot, fit_name_to_plot = select_valid_fits(
        fit_output,
        subset_plot=subset_plot,
        remove_lower=remove_lower,
        remove_higher=remove_higher,
    )

    figsize = utils.return_key(kwargs, "figsize", (10, 10))

    if plot:
        fig, ax = plt.subplots(len(parameters), 1, figsize=figsize, sharex=True)
    param_dict, error_dict = {}, {}
    for j, param_name in enumerate(parameters):
        param_dict[param_name] = []
        error_dict[param_name] = []
    for i, fit in enumerate(fit_to_plot):
        for j, param_name in enumerate(parameters):
            if (param_name == "fs8") & (compute_fs8_from_beta):
                param = fit[0]["beta_f"] * fit[0]["bs8"]
                error = (
                    fit[0]["beta_f"]
                    * fit[0]["bs8"]
                    * np.sqrt(
                        (fit[2]["bs8"] / fit[0]["bs8"]) ** 2
                        + (fit[2]["beta_f"] / fit[0]["beta_f"]) ** 2
                    )
                )
            else:
                param = fit[0][param_name]
                error = fit[2][param_name]
            param_dict[param_name].append(param)
            error_dict[param_name].append(error)
            if plot:
                ax[j].errorbar(
                    i,
                    param,
                    error,
                    marker=".",
                    ls="None",
                    color="C1",
                )

                ax[j].set_ylabel(param_name, fontsize=18)

                if fiducials is not None:
                    if fiducials[j] is not None:
                        ax[j].axhline(fiducials[j], ls=":", color="k")
    if plot:
        ax[0].margins(x=0.005)
        fig.tight_layout()
    return fit_name_to_plot, param_dict, error_dict


def plot_all_mean_fits(
    fit_output,
    parameters,
    fiducials=None,
    weighted_mean=True,
    compute_fs8_from_beta=False,
    plot_std_error=False,
    plot_error_bar_of_mean=True,
    subset_plot=None,
    remove_lower=None,
    remove_higher=None,
    plot=True,
    use_minos=False,
    **kwargs,
):
    """Plot mean parameter values and errors across grouped fits.

    Groups fits by a substring in filenames, computes mean and error metrics, and
    optionally plots both means and errors.

    Args:
        fit_output (str): Directory containing pickled fit outputs.
        parameters (list[str]): Parameters to summarize.
        fiducials (list[float]|None): Reference lines per parameter.
        weighted_mean (bool): Weight by inverse Hessian variance when True.
        compute_fs8_from_beta (bool): Use `fs8 = beta_f * bs8`.
        plot_std_error (bool): Plot standard deviation instead of mean error.
        plot_error_bar_of_mean (bool): Plot error-of-the-mean instead of mean error.
        subset_plot (list[str]|None): Only include fits matching substrings.
        remove_lower (dict|None): Exclude fits where param < threshold.
        remove_higher (dict|None): Exclude fits where param > threshold.
        plot (bool): Whether to render plots.
        use_minos (bool): Use MINOS errors if available; fall back to Hessian.
        **kwargs: Matplotlib options, e.g., `figsize`.

    Returns:
        tuple: `(unique_fit_prop, mean_param_dict, mean_error_dict, error_mean_dict, std_dict, count_dict)`.
    """

    fit_to_plot, fit_name_to_plot = select_valid_fits(
        fit_output,
        subset_plot=subset_plot,
        remove_lower=remove_lower,
        remove_higher=remove_higher,
    )

    figsize = utils.return_key(kwargs, "figsize", (10, 10))

    fit_prop = []
    for i in range(len(fit_name_to_plot)):
        fit_prop.append(
            fit_name_to_plot[i].split("fitted_parameters_")[-1].split("_box")[0]
        )

    fit_prop = np.array(fit_prop)
    unique_fit_prop = np.sort(np.unique(fit_prop))
    if plot:
        fig, ax = plt.subplots(len(parameters), 1, figsize=figsize, sharex=True)
        fig2, ax2 = plt.subplots(len(parameters), 1, figsize=figsize, sharex=True)

    text = []
    mean_param_dict, mean_error_dict, error_mean_dict, std_dict, count_dict = (
        {},
        {},
        {},
        {},
        {},
    )
    for j, param_name in enumerate(parameters):
        mean_param_dict[param_name] = []
        mean_error_dict[param_name] = []
        error_mean_dict[param_name] = []
        std_dict[param_name] = []
        count_dict[param_name] = []

    for i, fit_p in enumerate(unique_fit_prop):

        mask = fit_prop == fit_p
        fits = np.array(fit_to_plot, dtype=object)[mask]

        for j, param_name in enumerate(parameters):
            if (param_name == "fs8") & (compute_fs8_from_beta):
                params = [
                    fits[i][0]["beta_f"] * fits[i][0]["bs8"] for i in range(len(fits))
                ]
                errors_hesse = [
                    (
                        fits[i][0]["beta_f"]
                        * fits[i][0]["bs8"]
                        * np.sqrt(
                            (fits[i][2]["bs8"] / fits[i][0]["bs8"]) ** 2
                            + (fits[i][2]["beta_f"] / fits[i][0]["beta_f"]) ** 2
                        )
                    )
                    for i in range(len(fits))
                ]
                if use_minos:
                    names = [fits[0][3][i].name for i in range(len(fits[0][3]))]
                    index_bs8 = np.argwhere(np.array(names) == "bs8")[0][0]
                    index_beta_f = np.argwhere(np.array(names) == "beta_f")[0][0]
                    for i in range(len(fits)):
                        error_bs8_low = fits[i][2]["bs8"]
                        error_bs8_high = fits[i][2]["bs8"]
                        error_betaf_low = fits[i][2]["beta_f"]
                        error_betaf_high = fits[i][2]["beta_f"]
                        try:
                            error_bs8_low = fits[i][3][index_bs8].lower
                            error_bs8_high = fits[i][3][index_bs8].upper
                            error_betaf_low = fits[i][3][index_beta_f].lower
                            error_betaf_high = fits[i][3][index_beta_f].upper
                        except IndexError:
                            print("Minos failed, taking hessian error")
                    errors = [
                        [
                            fits[i][0]["beta_f"]
                            * fits[i][0]["bs8"]
                            * np.sqrt(
                                (error_bs8_low / fits[i][0]["bs8"]) ** 2
                                + (error_betaf_low / fits[i][0]["beta_f"]) ** 2
                            )
                            for i in range(len(fits))
                        ],
                        [
                            fits[i][0]["beta_f"]
                            * fits[i][0]["bs8"]
                            * np.sqrt(
                                (error_bs8_high / fits[i][0]["bs8"]) ** 2
                                + (error_betaf_high / fits[i][0]["beta_f"]) ** 2
                            )
                            for i in range(len(fits))
                        ],
                    ]
                else:
                    errors = errors_hesse
            else:
                params = [fits[i][0][param_name] for i in range(len(fits))]
                errors_hesse = [fits[i][2][param_name] for i in range(len(fits))]
                if use_minos:
                    errors = [[], []]
                    for i in range(len(fits)):
                        try:
                            names = [fits[i][3][j].name for j in range(len(fits[0][3]))]
                            index = np.argwhere(np.array(names) == param_name)[0][0]
                            errors[0].append(abs(fits[i][3][index].lower))
                            errors[1].append(abs(fits[i][3][index].upper))
                        except IndexError:
                            errors[0].append(fits[i][2][param_name])
                            errors[1].append(fits[i][2][param_name])
                else:
                    errors = errors_hesse
            if weighted_mean:
                weigths_errors = [1 / (error**2) for error in errors_hesse]
                mean_param = np.average(params, weights=weigths_errors)
            else:
                mean_param = np.mean(params)
            if use_minos:
                error_mean_param = [
                    [np.mean(errors[0]) / np.sqrt(len(mask[mask]))],
                    [np.mean(errors[1]) / np.sqrt(len(mask[mask]))],
                ]
                mean_error_param = [
                    [np.mean(errors[0])],
                    [np.mean(errors[1])],
                ]
            else:
                error_mean_param = np.mean(errors) / np.sqrt(len(mask[mask]))
                mean_error_param = np.mean(errors)
            std_param = np.std(params)
            count = len(params)

            mean_param_dict[param_name].append(np.array(mean_param))
            mean_error_dict[param_name].append(np.array(mean_error_param))
            error_mean_dict[param_name].append(np.array(error_mean_param))
            std_dict[param_name].append(np.array(std_param))
            count_dict[param_name].append(np.array(count))
            if plot:
                if plot_std_error:
                    if plot_error_bar_of_mean:
                        error_plot = std_param / np.sqrt(len(mask[mask]))
                    else:
                        error_plot = std_param
                else:
                    if plot_error_bar_of_mean:
                        error_plot = error_mean_param
                    else:
                        error_plot = mean_error_param
                ax[j].errorbar(
                    i, mean_param, error_plot, marker=".", ls="None", color="C1"
                )

                ax[j].set_ylabel(param_name, fontsize=18)

                ax2[j].plot(i, error_plot, marker=".", ls="None", color="C1")
                ax2[j].set_ylabel(r"$\sigma$(" + param_name + ")", fontsize=18)

                if fiducials is not None:
                    if fiducials[j] is not None:
                        ax[j].axhline(fiducials[j], ls=":", color="k")

        text.append(fit_p)

    j_index = np.arange(len(unique_fit_prop))
    if plot:
        ax[-1].set_xticks(j_index, np.array(text), rotation=90, fontsize=10)
        ax[0].margins(x=0.005)
        fig.tight_layout()

        ax2[-1].set_xticks(j_index, np.array(text), rotation=90, fontsize=10)
        ax2[0].margins(x=0.005)
        fig2.tight_layout()

    return (
        unique_fit_prop,
        mean_param_dict,
        mean_error_dict,
        error_mean_dict,
        std_dict,
        count_dict,
    )


def select_valid_fits(
    fit_output,
    subset_plot=None,
    remove_lower=None,
    remove_higher=None,
):
    """Select valid fit files based on status flags and filters.

    Args:
        fit_output (str): Directory of pickled fit outputs.
        subset_plot (list[str]|None): Only include filenames containing these substrings.
        remove_lower (dict|None): Exclude fits where param < threshold.
        remove_higher (dict|None): Exclude fits where param > threshold.

    Returns:
        tuple[list, list]: Fit objects and their filenames.
    """
    all_fit = glob.glob(os.path.join(fit_output, "*"))

    fit_to_plot = []
    fit_name_to_plot = []
    for f in all_fit:
        valid_fit = True
        if subset_plot is not None:
            for subset in subset_plot:
                if subset not in f:
                    valid_fit = False
        if valid_fit:
            fit = pickle.load(open(f, "rb"))
            if fit[3] is False:
                valid_fit = False
            elif fit[4] is False:
                valid_fit = False
            if remove_lower is not None:
                for param in remove_lower.keys():
                    if fit[0][param] < remove_lower[param]:
                        valid_fit = False
            if remove_higher is not None:
                for param in remove_higher.keys():
                    if fit[0][param] > remove_higher[param]:
                        valid_fit = False
        if valid_fit:
            fit_to_plot.append(fit)
            fit_name_to_plot.append(f)
    return fit_to_plot, fit_name_to_plot
