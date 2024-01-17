import matplotlib.pyplot as plt
import numpy as np

from flip import utils
from flip.covariance import cov_utils


def plot_1d_contraction(
    contraction,
    parameter_dict,
    rs_multiplied=True,
):
    contraction_sum = contraction.compute_contraction_sum(parameter_dict)
    coord = contraction.coordinates_dict

    index_min_perpendicular = np.argmin(np.abs(coord["r_perpendicular"][:, 0]))
    index_min_parallel = np.argmin(np.abs(coord["r_parallel"][0, :]))

    _, ax = plt.subplots(1, 3, figsize=(17, 5))

    if contraction.model_type in ["density", "density_velocity", "full"]:
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

    if contraction.model_type == "full":
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

    if contraction.model_type in ["velocity", "density_velocity", "full"]:
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

    if contraction.model_type in ["density", "density_velocity", "full"]:
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

    if contraction.model_type == "full":
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

    if contraction.model_type in ["velocity", "density_velocity", "full"]:
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
    **kwargs,
):
    vmin = utils.return_key(kwargs, "vmin", -0.1)
    vmax = utils.return_key(kwargs, "vmax", 0.1)

    parameter_names = likelihood.parameter_names
    parameter_values = [
        parameter_dict[parameters]["value"] for parameters in parameter_names
    ]
    parameter_values_dict = dict(zip(parameter_names, parameter_values))

    _, vector_error = likelihood.load_data_vector(
        likelihood.covariance.model_type,
        parameter_values_dict,
    )
    covariance_sum = likelihood.covariance.compute_covariance_sum(
        parameter_values_dict, vector_error
    )

    correlation_sum = cov_utils.return_correlation_matrix(covariance_sum)
    plt.imshow(correlation_sum, vmin=vmin, vmax=vmax)
    plt.colorbar()
