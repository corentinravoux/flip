import numpy as np
import scipy


def M_gg_0_0_0(sig_g):
    def func(k):
        return (
            (1 / 2)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
        )

    return func


def N_gg_0_0_0(theta, phi):
    return 1


def M_gg_0_0_1(sig_g):
    def func(k):
        return (
            (1 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            + (3 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (9 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_0_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_0_2_0(sig_g):
    def func(k):
        return (
            -1
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 3
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (3 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
        )

    return func


def N_gg_0_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_0_2_1(sig_g):
    def func(k):
        return (
            -1
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 3
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (3 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
        )

    return func


def N_gg_0_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_0_2_2(sig_g):
    def func(k):
        return (
            (1 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            + (3 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (9 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_0_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_0_4_0(sig_g):
    def func(k):
        return (
            (1 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            + (3 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (9 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_0_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_0_0(sig_g):
    def func(k):
        return -np.sqrt(2) * np.sqrt(np.pi) * np.exp(
            -1 / 2 * k**2 * sig_g**2
        ) * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) / (
            k**3 * sig_g**3
        ) + np.pi * scipy.special.erf(
            (1 / 2) * np.sqrt(2) * k * sig_g
        ) ** 2 / (
            k**4 * sig_g**4
        )

    return func


def N_gg_1_0_0(theta, phi):
    return 1


def M_gg_1_0_1(sig_g):
    def func(k):
        return (
            (1 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (1 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + 3 * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (3 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (27 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 27
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (27 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_1_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_1_2_0(sig_g):
    def func(k):
        return (
            -1
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 1
            / 2
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (3 / 2) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + 3
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_1_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_1_2_1(sig_g):
    def func(k):
        return (
            -1
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 1
            / 2
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (3 / 2) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + 3
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_1_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_1_2_2(sig_g):
    def func(k):
        return (
            (1 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (1 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + 3 * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (3 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (27 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 27
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (27 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_1_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_1_4_0(sig_g):
    def func(k):
        return (
            (1 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (1 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + 3 * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (3 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (27 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 27
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (27 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_1_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_0_0(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 2)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_2_0_0(theta, phi):
    return 1


def M_gg_2_0_1(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + 9 * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 9
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (81 / 4) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 81
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (81 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_2_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_2_0(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 1
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 1
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (9 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (9 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_2_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_2_2_1(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 1
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 1
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (9 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (9 / 4)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_2_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_2_2_2(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + 9 * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 9
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (81 / 4) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 81
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (81 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_2_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_2_4_0(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + 9 * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 9
            / 4
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (81 / 4) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 81
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (81 / 8)
            * np.pi
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_2_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gv_0_1_0(sig_g):
    def func(k):
        return (
            (1 / 6)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**2 * sig_g)
        )

    return func


def N_gv_0_1_0(theta, phi):
    return -3 * np.cos(phi - 1 / 2 * theta)


def M_gv_0_1_1(sig_g):
    def func(k):
        return (
            -1
            / 12
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**2 * sig_g)
            - 1 / 2 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            + (1 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
        )

    return func


def N_gv_0_1_1(theta, phi):
    return -3 / 2 * np.cos(phi - 1 / 2 * theta) - 9 / 2 * np.cos(phi + (3 / 2) * theta)


def M_gv_0_3_0(sig_g):
    def func(k):
        return (
            -1
            / 12
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**2 * sig_g)
            - 1 / 2 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            + (1 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
        )

    return func


def N_gv_0_3_0(theta, phi):
    return (
        -9 / 4 * np.cos(phi - 1 / 2 * theta)
        - 9 / 8 * np.cos(phi + (3 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi + (1 / 2) * theta)
    )


def M_gv_1_1_0(sig_g):
    def func(k):
        return -1 / 3 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2) + (
            1 / 6
        ) * np.sqrt(2) * np.sqrt(np.pi) * scipy.special.erf(
            (1 / 2) * np.sqrt(2) * k * sig_g
        ) / (
            k**4 * sig_g**3
        )

    return func


def N_gv_1_1_0(theta, phi):
    return -3 * np.cos(phi - 1 / 2 * theta)


def M_gv_1_1_1(sig_g):
    def func(k):
        return (
            -1 / 3 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 1
            / 12
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 3 / 2 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            + (3 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def N_gv_1_1_1(theta, phi):
    return -3 / 2 * np.cos(phi - 1 / 2 * theta) - 9 / 2 * np.cos(phi + (3 / 2) * theta)


def M_gv_1_3_0(sig_g):
    def func(k):
        return (
            -1 / 3 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 1
            / 12
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 3 / 2 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            + (3 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * scipy.special.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def N_gv_1_3_0(theta, phi):
    return (
        -9 / 4 * np.cos(phi - 1 / 2 * theta)
        - 9 / 8 * np.cos(phi + (3 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi + (1 / 2) * theta)
    )


def M_vv_0_0_0(sig_g):
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_0_2_0(sig_g):
    def func(k):
        return (1 / 9) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


dictionary_terms = {
    "gg_0_0": 2,
    "gg_0_1": 0,
    "gg_0_2": 3,
    "gg_0_3": 0,
    "gg_0_4": 1,
    "gg_1_0": 2,
    "gg_1_1": 0,
    "gg_1_2": 3,
    "gg_1_3": 0,
    "gg_1_4": 1,
    "gg_2_0": 2,
    "gg_2_1": 0,
    "gg_2_2": 3,
    "gg_2_3": 0,
    "gg_2_4": 1,
    "gv_0_0": 0,
    "gv_0_1": 2,
    "gv_0_2": 0,
    "gv_0_3": 1,
    "gv_1_0": 0,
    "gv_1_1": 2,
    "gv_1_2": 0,
    "gv_1_3": 1,
    "vv_0_0": 1,
    "vv_0_1": 0,
    "vv_0_2": 1,
}
