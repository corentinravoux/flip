def get_coefficients(model_type, parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["vv"] = [parameter_values_dict["fs8"] ** 2]
    return coefficients_dict


def get_diagonal_coefficients(model_type, parameter_values_dict):
    coefficients_dict = {}
    coefficients_dict["vv"] = parameter_values_dict["sigv"] ** 2
    return coefficients_dict


def get_vector_coefficients():
    alpha = par["alpha"]
    beta = par["beta"]
    M0 = par["M0"]
    sigM = par["sigM"]

    mu, cov_dmu = compute_mu(
        df.mb,
        df.x1,
        df.c,
        df.e_mb,
        df.e_x1,
        df.e_c,
        df.cov_mb_x1,
        df.cov_mb_c,
        df.cov_x1_c,
        alpha,
        beta,
        M0,
    )
    cov_dmu += sigM**2

    muth = (
        5 * np.log10((1 + df["zobs"]) * cosmo.comoving_distance(df["zobs"]).value) + 25
    )
    dmu = mu - muth

    pfct = _C_LIGHT_KMS_ * np.log(10) / 5
    if vel_est == 0:
        zdep = df[zkey] / (1 + df[zkey])
    elif vel_est == 1:
        zdep = ((1 + df[zkey]) / df[zkey] - 1.0) ** (-1)
    elif vel_est == 2:
        zdep = (
            (1 + df[zkey])
            * _C_LIGHT_KMS_
            / (cosmo.H(df[zkey]) * cosmo.comoving_distance(df[zkey]).value)
            - 1.0
        ) ** (-1)

    vest = -pfct * zdep * dmu
    verr = pfct * zdep * np.sqrt(cov_dmu)

    return
