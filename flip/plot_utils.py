import matplotlib.pyplot as plt
import numpy as np


def plot_2d_contraction(
    covariance,
    parameter_dict,
    rs_multiplied=True,
    rmin=30,
):
    """
    The plot_2d_contraction function plots the 2D correlation function for a given covariance model.

    Args:
        covariance: Compute the contraction sum
        parameter_dict: Pass in the values of the parameters that are used to compute the covariance
        rs_multiplied: Multiply the correlation function by r^2
        rmin: Mask the plot for small scales

    Returns:
        A plot of the 2d contraction
    """
    contraction_sum = covariance.compute_contraction_sum(parameter_dict)
    coord = covariance.contraction_coordinates_dict

    r_perpendicular_min = np.min(coord["r_perpendicular"])
    r_perpendicular_max = np.max(coord["r_perpendicular"])
    r_parallel_min = np.min(coord["r_parallel"])
    r_parallel_max = np.max(coord["r_parallel"])
    extent = [
        r_parallel_min,
        r_parallel_max,
        r_perpendicular_max,
        r_perpendicular_min,
    ]

    _, ax = plt.subplots(1, 3, figsize=(17, 5))

    if covariance.model_type in ["density", "density_velocity", "full"]:
        xi_gg = contraction_sum["gg"]
        mask_rmin = np.where(coord["r"] > rmin)
        xi_gg_plot = np.full(xi_gg.shape, np.nan)
        xi_gg_plot[mask_rmin] = xi_gg[mask_rmin]

        ax_plot = ax[0]
        if rs_multiplied:
            image = ax_plot.imshow(
                coord["r"] ** 2 * xi_gg_plot,
                extent=extent,
            )
            ax_plot.set_title(r"$r^2 C_{gg}(r)$", fontsize=15)
        else:
            ax_plot.imshow(
                xi_gg_plot,
                extent=extent,
            )
            ax_plot.set_title(r"$C_{gg}(r)$", fontsize=15)

        plt.colorbar(image, ax=ax_plot)
        ax_plot.set_xlabel(r"$r_{\bot}$")
        ax_plot.set_ylabel(r"$r_{\parallel}$")
        ax_plot.set_title(r"$r^2 C_{gg}(r)$", fontsize=15)

    if covariance.model_type in ["velocity", "density_velocity", "full"]:
        xi_vv = contraction_sum["vv"]
        ax_plot = ax[2]

        image = ax_plot.imshow(
            xi_vv,
            extent=extent,
        )
        plt.colorbar(image, ax=ax_plot)
        ax_plot.set_xlabel(r"$r_{\bot}$")
        ax_plot.set_ylabel(r"$r_{\parallel}$")
        ax_plot.set_title(r"$C_{vv}(r)$", fontsize=15)

    if covariance.model_type == "full":
        xi_gv = contraction_sum["gv"]
        ax_plot = ax[1]

        image = ax_plot.imshow(
            xi_gv,
            extent=extent,
        )
        plt.colorbar(image, ax=ax_plot)
        ax_plot.set_xlabel(r"$r_{\bot}$")
        ax_plot.set_ylabel(r"$r_{\parallel}$")
        ax_plot.set_title(r"$C_{gv}(r)$", fontsize=15)
