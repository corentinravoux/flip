import numpy as np


def K_gg_0_0(theta, phi, sig_g):
    def func(k):
        return (1 / 2) * np.sqrt(np.pi) * np.erf(k * sig_g) / (k * sig_g)

    return func


def K_gg_0_1(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_0_2(theta, phi, sig_g):
    def func(k):
        return (
            (5 / 16)
            * (3 * np.cos(phi) ** 2 - 1)
            * (
                -2
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                - 6 * k * sig_g
                + 3 * np.sqrt(np.pi) * np.exp(k**2 * sig_g**2) * np.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**3 * sig_g**3)
        )

    return func


def K_gg_0_3(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_0_4(theta, phi, sig_g):
    def func(k):
        return (
            (9 / 512)
            * (
                12
                * np.sqrt(np.pi)
                * k**4
                * sig_g**4
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                + 20
                * k**3
                * sig_g**3
                * (-35 * np.cos(phi) ** 4 + 30 * np.cos(phi) ** 2 - 3)
                - 60
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                - 210 * k * sig_g * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                + 105
                * np.sqrt(np.pi)
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**5 * sig_g**5)
        )

    return func


def K_gg_1_0(theta, phi, sig_g):
    def func(k):
        return -1 / 2 * np.exp(-(k**2) * sig_g**2) / (k**2 * sig_g**2) + (
            1 / 4
        ) * np.sqrt(np.pi) * np.erf(k * sig_g) / (k**3 * sig_g**3)

    return func


def K_gg_1_1(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_1_2(theta, phi, sig_g):
    def func(k):
        return (
            (5 / 32)
            * (
                8 * k**3 * sig_g**3 * (1 - 3 * np.cos(phi) ** 2)
                - 2
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (3 * np.cos(phi) ** 2 - 1)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                - 18 * k * sig_g * (3 * np.cos(phi) ** 2 - 1)
                + 9
                * np.sqrt(np.pi)
                * (3 * np.cos(phi) ** 2 - 1)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**5 * sig_g**5)
        )

    return func


def K_gg_1_3(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_1_4(theta, phi, sig_g):
    def func(k):
        return (
            (9 / 1024)
            * (
                64
                * k**5
                * sig_g**5
                * (-35 * np.cos(phi) ** 4 + 30 * np.cos(phi) ** 2 - 3)
                + 12
                * np.sqrt(np.pi)
                * k**4
                * sig_g**4
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                + 340
                * k**3
                * sig_g**3
                * (-35 * np.cos(phi) ** 4 + 30 * np.cos(phi) ** 2 - 3)
                - 180
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                - 1050 * k * sig_g * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                + 525
                * np.sqrt(np.pi)
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**7 * sig_g**7)
        )

    return func


def K_gg_2_0(theta, phi, sig_g):
    def func(k):
        return (
            (1 / 8)
            * (
                -4 * k**3 * sig_g**3
                - 6 * k * sig_g
                + 3 * np.sqrt(np.pi) * np.exp(k**2 * sig_g**2) * np.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**5 * sig_g**5)
        )

    return func


def K_gg_2_1(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_2_2(theta, phi, sig_g):
    def func(k):
        return (
            (5 / 64)
            * (
                16 * k**5 * sig_g**5 * (1 - 3 * np.cos(phi) ** 2)
                + 48 * k**3 * sig_g**3 * (1 - 3 * np.cos(phi) ** 2)
                - 6
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (3 * np.cos(phi) ** 2 - 1)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                - 90 * k * sig_g * (3 * np.cos(phi) ** 2 - 1)
                + 45
                * np.sqrt(np.pi)
                * (3 * np.cos(phi) ** 2 - 1)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**7 * sig_g**7)
        )

    return func


def K_gg_2_3(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_2_4(theta, phi, sig_g):
    def func(k):
        return (
            (9 / 2048)
            * (
                128
                * k**7
                * sig_g**7
                * (-35 * np.cos(phi) ** 4 + 30 * np.cos(phi) ** 2 - 3)
                + 832
                * k**5
                * sig_g**5
                * (-35 * np.cos(phi) ** 4 + 30 * np.cos(phi) ** 2 - 3)
                + 36
                * np.sqrt(np.pi)
                * k**4
                * sig_g**4
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                + 3100
                * k**3
                * sig_g**3
                * (-35 * np.cos(phi) ** 4 + 30 * np.cos(phi) ** 2 - 3)
                - 900
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
                - 7350 * k * sig_g * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                + 3675
                * np.sqrt(np.pi)
                * (35 * np.cos(phi) ** 4 - 30 * np.cos(phi) ** 2 + 3)
                * np.exp(k**2 * sig_g**2)
                * np.erf(k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            / (k**9 * sig_g**9)
        )

    return func


def K_gv_0_0(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gv_0_1(theta, phi, sig_g):
    def func(k):
        return -3 * np.exp(-1 / 2 * k**2 * sig_g**2) * np.cos(phi) / (
            k**3 * sig_g**2
        ) + (3 / 2) * np.sqrt(2) * np.sqrt(np.pi) * np.cos(phi) * np.erf(
            (1 / 2) * np.sqrt(2) * k * sig_g
        ) / (
            k**4 * sig_g**3
        )

    return func


def K_gv_0_2(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gv_0_3(theta, phi, sig_g):
    def func(k):
        return (
            -7
            / 8
            * (5 * np.cos(phi) ** 2 - 3)
            * (
                4 * k**3 * sig_g**3 * np.exp((1 / 2) * k**2 * sig_g**2)
                + 3
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * np.exp(k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 30 * k * sig_g * np.exp((1 / 2) * k**2 * sig_g**2)
                - 15
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * np.exp(k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
            * np.cos(phi)
            / (k**6 * sig_g**5)
        )

    return func


def K_gv_0_4(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gv_1_0(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gv_1_1(theta, phi, sig_g):
    def func(k):
        return (
            -3
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * np.cos(phi)
            / (k**3 * sig_g**2)
            - 9
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * np.cos(phi)
            / (k**5 * sig_g**4)
            + (9 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.cos(phi)
            * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def K_gv_1_2(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gv_1_3(theta, phi, sig_g):
    def func(k):
        return (
            -7
            / 8
            * (5 * np.cos(phi) ** 2 - 3)
            * (
                4 * k**5 * sig_g**5 * np.exp(k**2 * sig_g**2)
                + 32 * k**3 * sig_g**3 * np.exp(k**2 * sig_g**2)
                + 9
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * np.exp((3 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 150 * k * sig_g * np.exp(k**2 * sig_g**2)
                - 75
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * np.exp((3 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            )
            * np.exp(-3 / 2 * k**2 * sig_g**2)
            * np.cos(phi)
            / (k**8 * sig_g**7)
        )

    return func


def K_gv_1_4(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_vv_0_0(theta, phi, sig_g):
    def func(k):
        return (1 / 3) / k**2

    return func


def K_vv_0_1(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_vv_0_2(theta, phi, sig_g):
    def func(k):
        return (np.cos(phi) ** 2 - 1 / 3) / k**2

    return func


def K_vv_0_3(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_vv_0_4(theta, phi, sig_g):
    def func(k):
        return 0

    return func
