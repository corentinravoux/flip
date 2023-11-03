import numpy as np
import itertools
import matplotlib.pyplot as plt


def build_inputs(
    ptt_file_name,
    pmt_file_name,
    pmm_file_name,
    qmax,
    pmax,
    sigmau,
    add_grid_window=True,
    grid_size=None,
    grid_kind=None,
    add=False,
    pmm_file_name_add=None,
):
    ktt = np.loadtxt(ptt_file_name)[0]
    kmt = np.loadtxt(pmt_file_name)[0]
    kmm = np.loadtxt(pmm_file_name)[0]

    ptt = np.loadtxt(ptt_file_name)[1]
    pmt = np.loadtxt(pmt_file_name)[1]
    pmm = np.loadtxt(pmm_file_name)[1]

    def Du(k, sigmau):
        return np.sin(k * sigmau) / (k * sigmau)

    if add:
        kmm_add = np.loadtxt(pmm_file_name_add)[0]
        pmm_add = np.loadtxt(pmm_file_name_add)[1]
    else:
        kmm_add = None
        pmm_gg_add = None

    if add_grid_window:
        grid_window_m_mm = grid_utils.compute_grid_window(
            grid_size, kmm, kind=grid_kind
        )
        grid_window_m_mt = grid_utils.compute_grid_window(
            grid_size, kmt, kind=grid_kind
        )

        if add:
            grid_window_m_mm_add = grid_utils.compute_grid_window(
                grid_size, kmm_add, kind=grid_kind
            )

    ptt_gg = ptt
    if add_grid_window:
        pmt_gg = pmt * grid_window_m_mt
        pmm_gg = pmm * grid_window_m_mm**2
        if add:
            pmm_gg_add = pmm_add * grid_window_m_mm_add**2
    else:
        pmt_gg = pmt
        pmm_gg = pmm
        if add:
            pmm_gg_add = pmm_add

    ptt_gv = ptt * (Du(ktt, sigmau))
    if add_grid_window:
        pmt_gv = pmt * (Du(kmt, sigmau)) * grid_window_m_mt
    else:
        pmt_gv = pmt * (Du(kmt, sigmau))

    ptt_vv = ptt * (Du(ktt, sigmau)) ** 2

    p_index_gg = np.arange(pmax + 1)
    q_index_gg = np.arange(qmax + 1)
    m_index_gg = np.arange(0, 2 * (qmax + pmax) + 1, 2)
    iter_pq_gg = np.array(list(itertools.product(p_index_gg, q_index_gg)))
    sum_iter_pq_gg = 2 * np.sum(iter_pq_gg, axis=1)

    p_index_gv = np.arange(pmax + 1)
    m_index_gv = 2 * p_index_gv

    return (
        kmm,
        kmt,
        ktt,
        pmm_gg,
        pmt_gg,
        ptt_gg,
        pmt_gv,
        ptt_gv,
        ptt_vv,
        ptt_gv,
        ptt_vv,
        kmm_add,
        pmm_gg_add,
        m_index_gg,
        iter_pq_gg,
        sum_iter_pq_gg,
        p_index_gv,
        m_index_gv,
    )


def build_2d_array(bin_centers_2d, r0):
    coord_rper_rpar = np.array(
        np.meshgrid(bin_centers_2d, bin_centers_2d, indexing="ij")
    ).reshape((2, len(bin_centers_2d) * len(bin_centers_2d)))
    # rpar 1, rperp 0

    coord_2d = np.zeros((3, len(bin_centers_2d) * len(bin_centers_2d)))
    coord_2d[0, :] = np.sqrt(coord_rper_rpar[0, :] ** 2 + coord_rper_rpar[1, :] ** 2)
    coord_2d[1, :] = np.arctan2(coord_rper_rpar[0, :], r0 + coord_rper_rpar[1, :])
    coord_2d[2, :] = np.arcsin(
        np.clip(
            (
                (r0 / coord_2d[0, :])
                + (coord_rper_rpar[0, :] / (coord_2d[0, :] * np.sin(coord_2d[1, :])))
            )
            * np.sqrt((1 - np.cos(coord_2d[1, :])) / 2),
            -1,
            1,
        )
    )

    return coord_2d


def compute_xi_lai_22_2d(
    params,
    bin_centers_2d,
    kmm,
    kmt,
    ktt,
    pmm_gg,
    pmt_gg,
    ptt_gg,
    pmt_gv,
    ptt_gv,
    ptt_vv,
    m_index_gg,
    iter_pq_gg,
    sum_iter_pq_gg,
    p_index_gv,
    m_index_gv,
    coord,
    sig_damp_mm_gg_m=None,
    hankel=True,
    add=False,
    kmm_add=None,
    pmm_gg_add=None,
):
    fs8 = params[0]
    bs8 = params[1]
    sig_g = params[2]
    if add:
        bs8_add = params[3]

    cov_gg_b2_m = []
    cov_gg_f2_m = []
    cov_gg_bf_m = []
    if add:
        cov_gg_add_m = []

    for m in m_index_gg:
        cov_gg_b2_m.append(
            joint_cov.coefficient_gg_b2_m(
                kmm,
                pmm_gg,
                iter_pq_gg,
                sum_iter_pq_gg,
                m,
                sig_damp_mm_gg_m,
                coord,
                hankel=hankel,
            )
        )
        cov_gg_f2_m.append(
            joint_cov.coefficient_gg_f2_m(
                ktt, ptt_gg, iter_pq_gg, sum_iter_pq_gg, m, coord, hankel=hankel
            )
        )
        cov_gg_bf_m.append(
            joint_cov.coefficient_gg_bf_m(
                kmt, pmt_gg, iter_pq_gg, sum_iter_pq_gg, m, coord, hankel=hankel
            )
        )
        if add:
            cov_gg_add_m.append(
                joint_cov.coefficient_gg_b2_m(
                    kmm_add,
                    pmm_gg_add,
                    iter_pq_gg,
                    sum_iter_pq_gg,
                    m,
                    sig_damp_mm_gg_m,
                    coord,
                    hankel=False,
                )
            )

    cov_gv_f2_m = []
    cov_gv_bf_m = []

    for p in p_index_gv:
        cov_gv_f2_m.append(joint_cov.coefficient_gv_f2_p(ktt, ptt_gv, p, coord))
        cov_gv_bf_m.append(joint_cov.coefficient_gv_bf_p(kmt, pmt_gv, p, coord))

    cov_vv = joint_cov.coefficient_vv(ktt, ptt_vv, coord)

    cov_gg_b2 = np.zeros((len(bin_centers_2d) * len(bin_centers_2d)))
    cov_gg_f2 = np.zeros((len(bin_centers_2d) * len(bin_centers_2d)))
    cov_gg_bf = np.zeros((len(bin_centers_2d) * len(bin_centers_2d)))
    if add:
        cov_gg_add = np.zeros((len(bin_centers_2d) * len(bin_centers_2d)))

    for i in range(len(m_index_gg)):
        cov_gg_b2 += cov_gg_b2_m[i] * sig_g ** m_index_gg[i]
        cov_gg_f2 += cov_gg_f2_m[i] * sig_g ** m_index_gg[i]
        cov_gg_bf += cov_gg_bf_m[i] * sig_g ** m_index_gg[i]
        if add:
            cov_gg_add += cov_gg_add_m[i] * sig_g ** m_index_gg[i]

    xi_gg = bs8**2 * cov_gg_b2 + bs8 * fs8 * cov_gg_bf + fs8**2 * cov_gg_f2
    if add:
        xi_gg = xi_gg + bs8_add**2 * cov_gg_add

    cov_gv_f2 = np.zeros((len(bin_centers_2d) * len(bin_centers_2d)))
    cov_gv_bf = np.zeros((len(bin_centers_2d) * len(bin_centers_2d)))

    for i in range(len(m_index_gv)):
        cov_gv_f2 += cov_gv_f2_m[i] * sig_g ** m_index_gv[i]
        cov_gv_bf += cov_gv_bf_m[i] * sig_g ** m_index_gv[i]

    xi_gv = 100 * bs8 * fs8 * cov_gv_bf + 100 * fs8**2 * cov_gv_f2

    xi_vv = 100**2 * fs8**2 * cov_vv

    return xi_gg, xi_gv, xi_vv


def evaluate_lai22(
    params,
    r0,
    ptt_file_name,
    pmt_file_name,
    pmm_file_name,
    qmax,
    pmax,
    sigmau,
    add_grid_window=True,
    grid_size=None,
    grid_kind=None,
    add=False,
    pmm_file_name_add=None,
    sig_damp_mm_gg_m=None,
    hankel=True,
):
    bin_centers_2d = np.linspace(1, 200, 200)

    coord_2d = build_2d_array(bin_centers_2d, r0)

    (
        kmm,
        kmt,
        ktt,
        pmm_gg,
        pmt_gg,
        ptt_gg,
        pmt_gv,
        ptt_gv,
        ptt_vv,
        ptt_gv,
        ptt_vv,
        kmm_add,
        pmm_gg_add,
        m_index_gg,
        iter_pq_gg,
        sum_iter_pq_gg,
        p_index_gv,
        m_index_gv,
    ) = build_inputs(
        ptt_file_name,
        pmt_file_name,
        pmm_file_name,
        qmax,
        pmax,
        sigmau,
        add_grid_window=add_grid_window,
        grid_size=grid_size,
        grid_kind=grid_kind,
        add=add,
        pmm_file_name_add=pmm_file_name_add,
    )

    xi_gg, xi_gv, xi_vv = compute_xi_lai_22_2d(
        params,
        bin_centers_2d,
        kmm,
        kmt,
        ktt,
        pmm_gg,
        pmt_gg,
        ptt_gg,
        pmt_gv,
        ptt_gv,
        ptt_vv,
        m_index_gg,
        iter_pq_gg,
        sum_iter_pq_gg,
        p_index_gv,
        m_index_gv,
        coord_2d,
        sig_damp_mm_gg_m=sig_damp_mm_gg_m,
        hankel=hankel,
        add=add,
        kmm_add=kmm_add,
        pmm_gg_add=pmm_gg_add,
    )

    return xi_gg, xi_gv, xi_vv, bin_centers_2d, coord_2d


def plot_lai22(xi_gg, xi_gv, xi_vv, bin_centers_2d, coord_2d, rs_multiplied=True):
    _, ax = plt.subplots(3, 3, figsize=(25, 22))

    if rs_multiplied:
        ax_plot = ax[0][0].imshow(
            coord_2d[0, :].reshape(len(bin_centers_2d), len(bin_centers_2d)) ** 2
            * xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d)),
            extent=[
                bin_centers_2d.min(),
                bin_centers_2d.max(),
                bin_centers_2d.max(),
                bin_centers_2d.min(),
            ],
        )
        ax[0][0].set_title(r"$r^2 C_{gg}(r)$", fontsize=15)
    else:
        ax_plot = ax[0][0].imshow(
            xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d)),
            extent=[
                bin_centers_2d.min(),
                bin_centers_2d.max(),
                bin_centers_2d.max(),
                bin_centers_2d.min(),
            ],
        )
        ax[0][0].set_title(r"$C_{gg}(r)$", fontsize=15)
    plt.colorbar(ax_plot, ax=ax[0][0])
    ax[0][0].set_xlabel(r"$r_{\bot}$")
    ax[0][0].set_ylabel(r"$r_{\parallel}$")
    ax[0][0].set_title(r"$r^2 C_{gg}(r)$", fontsize=15)
    if rs_multiplied:
        ax[1][0].plot(
            bin_centers_2d,
            bin_centers_2d**2
            * xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d))[0, :],
        )
        ax[1][0].plot(
            bin_centers_2d,
            bin_centers_2d**2
            * xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d))[:, 0],
        )
        ax[1][0].set_ylabel(r"$r_{i}^2 C_{gg}(r_{i})$", fontsize=15)
    else:
        ax[1][0].plot(
            bin_centers_2d,
            xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d))[0, :],
        )
        ax[1][0].plot(
            bin_centers_2d,
            xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d))[:, 0],
        )
        ax[1][0].set_ylabel(r"$C_{gg}(r_{i})$", fontsize=15)
    ax[1][0].set_ylabel(r"$r_{i}^2 C_{gg}(r_{i})$", fontsize=15)
    ax[1][0].set_xlabel(r"$r_{i}$", fontsize=15)
    ax[1][0].legend([r"$\parallel$", r"$\bot$"], fontsize=15)

    if rs_multiplied:
        ax[2][0].plot(
            bin_centers_2d,
            bin_centers_2d**2
            * (
                xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d))[0, :]
                - xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d))[:, 0]
            ),
        )
        ax[2][0].set_ylabel(
            r"$r_{\parallel / \bot}^{2} (C_{gg}(r_{\parallel}) - C_{gg}(r_{\bot}))$",
            fontsize=15,
        )
    else:
        ax[2][0].plot(
            bin_centers_2d,
            xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d))[0, :]
            - xi_gg.reshape(len(bin_centers_2d), len(bin_centers_2d))[:, 0],
        )
        ax[2][0].set_ylabel(r"$C_{gg}(r_{\parallel}) - C_{gg}(r_{\bot})$", fontsize=15)

    ax[2][0].set_xlabel(r"$r_{\parallel / \bot}[h^{-1}.\mathrm{Mpc}]$", fontsize=15)

    ax_plot = ax[0][1].imshow(
        xi_gv.reshape(len(bin_centers_2d), len(bin_centers_2d)),
        extent=[
            bin_centers_2d.min(),
            bin_centers_2d.max(),
            bin_centers_2d.max(),
            bin_centers_2d.min(),
        ],
    )
    plt.colorbar(ax_plot, ax=ax[0][1])
    ax[0][1].set_xlabel(r"$r_{\bot}$")
    ax[0][1].set_ylabel(r"$r_{\parallel}$")
    ax[0][1].set_title(r"$C_{gv}(r)$", fontsize=15)
    ax[1][1].plot(
        bin_centers_2d, xi_gv.reshape(len(bin_centers_2d), len(bin_centers_2d))[0, :]
    )
    ax[1][1].plot(
        bin_centers_2d, xi_gv.reshape(len(bin_centers_2d), len(bin_centers_2d))[:, 0]
    )
    ax[1][1].set_ylabel(r"$C_{gv}(r_{i})$", fontsize=15)
    ax[1][1].set_xlabel(r"$r_{i}$", fontsize=15)
    ax[1][1].legend([r"$\parallel$", r"$\bot$"], fontsize=15)
    ax[2][1].plot(
        bin_centers_2d,
        xi_gv.reshape(len(bin_centers_2d), len(bin_centers_2d))[0, :]
        - xi_gv.reshape(len(bin_centers_2d), len(bin_centers_2d))[:, 0],
    )
    ax[2][1].set_ylabel(r"$C_{gv}(r_{\parallel}) - C_{gv}(r_{\bot})$", fontsize=15)
    ax[2][1].set_xlabel(r"$r_{\parallel / \bot}[h^{-1}.\mathrm{Mpc}]$", fontsize=15)

    ax_plot = ax[0][2].imshow(
        xi_vv.reshape(len(bin_centers_2d), len(bin_centers_2d)),
        extent=[
            bin_centers_2d.min(),
            bin_centers_2d.max(),
            bin_centers_2d.max(),
            bin_centers_2d.min(),
        ],
    )
    plt.colorbar(ax_plot, ax=ax[0][2])
    ax[0][2].set_xlabel(r"$r_{\bot}$")
    ax[0][2].set_ylabel(r"$r_{\parallel}$")
    ax[0][2].set_title(r"$C_{vv}(r)$", fontsize=15)
    ax[1][2].plot(
        bin_centers_2d, xi_vv.reshape(len(bin_centers_2d), len(bin_centers_2d))[0, :]
    )
    ax[1][2].plot(
        bin_centers_2d, xi_vv.reshape(len(bin_centers_2d), len(bin_centers_2d))[:, 0]
    )
    ax[1][2].set_ylabel(r"$C_{vv}(r_{i})$", fontsize=15)
    ax[1][2].set_xlabel(r"$r_{i}$", fontsize=15)
    ax[1][2].legend([r"$\parallel$", r"$\bot$"], fontsize=15)
    ax[2][2].plot(
        bin_centers_2d,
        xi_vv.reshape(len(bin_centers_2d), len(bin_centers_2d))[0, :]
        - xi_vv.reshape(len(bin_centers_2d), len(bin_centers_2d))[:, 0],
    )
    ax[2][2].set_ylabel(r"$C_{vv}(r_{\parallel}) - C_{vv}(r_{\bot})$", fontsize=15)
    ax[2][2].set_xlabel(r"$r_{\parallel / \bot}[h^{-1}.\mathrm{Mpc}]$", fontsize=15)
