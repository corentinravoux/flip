import numpy as np
import scipy


def M_gg_0_0_0(sig_g):
    def func(k):
        return (1 / 2) * np.sqrt(np.pi) * scipy.special.erf(k * sig_g) / (k * sig_g)

    return func


def N_gg_0_0_0(theta, phi):
    return 1


def M_gg_0_2_0(sig_g):
    def func(k):
        return (
            -5 / 4 * np.sqrt(np.pi) * scipy.special.erf(k * sig_g) / (k * sig_g)
            - 15 / 4 * np.exp(-(k**2) * sig_g**2) / (k**2 * sig_g**2)
            + (15 / 8)
            * np.sqrt(np.pi)
            * scipy.special.erf(k * sig_g)
            / (k**3 * sig_g**3)
        )

    return func


def N_gg_0_2_0(theta, phi):
    return (3 / 4) * np.cos(2 * phi) + 1 / 4


def M_gg_0_4_0(sig_g):
    def func(k):
        return (
            (27 / 16) * np.sqrt(np.pi) * scipy.special.erf(k * sig_g) / (k * sig_g)
            - 45 / 16 * np.exp(-(k**2) * sig_g**2) / (k**2 * sig_g**2)
            - 135
            / 16
            * np.sqrt(np.pi)
            * scipy.special.erf(k * sig_g)
            / (k**3 * sig_g**3)
            - 945 / 32 * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (945 / 64)
            * np.sqrt(np.pi)
            * scipy.special.erf(k * sig_g)
            / (k**5 * sig_g**5)
        )

    return func


def N_gg_0_4_0(theta, phi):
    return (5 / 16) * np.cos(2 * phi) + (35 / 64) * np.cos(4 * phi) + 9 / 64


def M_gg_1_0_0(sig_g):
    def func(k):
        return -np.exp(-(k**2) * sig_g**2) / (k**2 * sig_g**2) + (
            1 / 2
        ) * np.sqrt(np.pi) * scipy.special.erf(k * sig_g) / (k**3 * sig_g**3)

    return func


def N_gg_1_0_0(theta, phi):
    return 1


def M_gg_1_2_0(sig_g):
    def func(k):
        return (
            -5 * np.exp(-(k**2) * sig_g**2) / (k**2 * sig_g**2)
            - 5 / 4 * np.sqrt(np.pi) * scipy.special.erf(k * sig_g) / (k**3 * sig_g**3)
            - 45 / 4 * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 8)
            * np.sqrt(np.pi)
            * scipy.special.erf(k * sig_g)
            / (k**5 * sig_g**5)
        )

    return func


def N_gg_1_2_0(theta, phi):
    return (3 / 4) * np.cos(2 * phi) + 1 / 4


def M_gg_1_4_0(sig_g):
    def func(k):
        return (
            -9
            / 64
            * (
                64 * k**5 * sig_g**5
                - 12
                * np.sqrt(np.pi)
                * k**4
                * sig_g**4
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
                + 340 * k**3 * sig_g**3
                + 180
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
                + 1050 * k * sig_g
                - 525
                * np.sqrt(np.pi)
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**7 * sig_g**7)
        )

    return func


def N_gg_1_4_0(theta, phi):
    return (5 / 16) * np.cos(2 * phi) + (35 / 64) * np.cos(4 * phi) + 9 / 64


def M_gg_2_0_0(sig_g):
    def func(k):
        return (
            -1
            / 8
            * (
                4 * k**3 * sig_g**3
                + 6 * k * sig_g
                - 3
                * np.sqrt(np.pi)
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**5 * sig_g**5)
        )

    return func


def N_gg_2_0_0(theta, phi):
    return 1


def M_gg_2_2_0(sig_g):
    def func(k):
        return (
            -5
            / 32
            * (
                16 * k**5 * sig_g**5
                + 48 * k**3 * sig_g**3
                + 6
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
                + 90 * k * sig_g
                - 45
                * np.sqrt(np.pi)
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**7 * sig_g**7)
        )

    return func


def N_gg_2_2_0(theta, phi):
    return (3 / 4) * np.cos(2 * phi) + 1 / 4


def M_gg_2_4_0(sig_g):
    def func(k):
        return (
            -9
            / 256
            * (
                128 * k**7 * sig_g**7
                + 832 * k**5 * sig_g**5
                - 36
                * np.sqrt(np.pi)
                * k**4
                * sig_g**4
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
                + 3100 * k**3 * sig_g**3
                + 900
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
                + 7350 * k * sig_g
                - 3675
                * np.sqrt(np.pi)
                * np.exp(k**2 * sig_g**2)
                * scipy.special.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**9 * sig_g**9)
        )

    return func


def N_gg_2_4_0(theta, phi):
    return (5 / 16) * np.cos(2 * phi) + (35 / 64) * np.cos(4 * phi) + 9 / 64


def M_gv_0_1_0(sig_g):
    def func(k):
        return -300 * np.exp(-1 / 2 * k**2 * sig_g**2) / (
            k**3 * sig_g**2
        ) + 150 * np.sqrt(2) * np.sqrt(np.pi) * scipy.special.erf(
            (1 / 2) * np.sqrt(2) * k * sig_g
        ) / (
            k**4 * sig_g**3
        )

    return func


def N_gv_0_1_0(theta, phi):
    return np.cos(phi)


def M_gv_0_3_0(sig_g):
    def func(k):
        return (
            -700 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 525
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 5250 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            + 2625
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def N_gv_0_3_0(theta, phi):
    return (3 / 8) * np.cos(phi) + (5 / 8) * np.cos(3 * phi)


def M_gv_1_1_0(sig_g):
    def func(k):
        return (
            -300 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 900 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            + 450
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def N_gv_1_1_0(theta, phi):
    return np.cos(phi)


def M_gv_1_3_0(sig_g):
    def func(k):
        return (
            -700 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 5600 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            - 1575
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
            - 26250 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**7 * sig_g**6)
            + 13125
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**8 * sig_g**7)
        )

    return func


def N_gv_1_3_0(theta, phi):
    return (3 / 8) * np.cos(phi) + (5 / 8) * np.cos(3 * phi)


def M_vv_0_0_0(sig_g):
    def func(k):
        return (10000 / 3) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 1


def M_vv_0_2_0(sig_g):
    def func(k):
        return (20000 / 3) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (3 / 4) * np.cos(2 * phi) + 1 / 4


dictionary_terms = {"gg": ["0", "1", "2"], "gv": ["0", "1"], "vv": ["0"]}
dictionary_lmax = {"gg": [4, 4, 4], "gv": [3, 3], "vv": [2]}
dictionary_subterms = {
    "gg_0_0": 1,
    "gg_0_1": 0,
    "gg_0_2": 1,
    "gg_0_3": 0,
    "gg_0_4": 1,
    "gg_1_0": 1,
    "gg_1_1": 0,
    "gg_1_2": 1,
    "gg_1_3": 0,
    "gg_1_4": 1,
    "gg_2_0": 1,
    "gg_2_1": 0,
    "gg_2_2": 1,
    "gg_2_3": 0,
    "gg_2_4": 1,
    "gv_0_0": 0,
    "gv_0_1": 1,
    "gv_0_2": 0,
    "gv_0_3": 1,
    "gv_1_0": 0,
    "gv_1_1": 1,
    "gv_1_2": 0,
    "gv_1_3": 1,
    "vv_0_0": 1,
    "vv_0_1": 0,
    "vv_0_2": 1,
}
multi_index_model = False
redshift_dependent_model = False
