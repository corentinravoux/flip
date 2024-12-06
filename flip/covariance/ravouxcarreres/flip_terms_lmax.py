import mpmath
import numpy
import scipy


def set_backend(module):
    global np, erf
    if module == "numpy":
        np = numpy
        erf = scipy.special.erf
    elif module == "mpmath":
        np = mpmath.mp
        erf = mpmath.erf


set_backend("numpy")

def M_gg_0_0_0(sig_g):
    def func(k):
        return (
            (1 / 2)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            + (3 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (9 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_0_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_0_0_2(sig_g):
    def func(k):
        return (
            (9 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 15
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 45
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (25 / 64) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 165
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (765 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (525 / 32) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (2625 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 1575
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (11025 / 64) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 11025
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (11025 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_0_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_0_2_0(sig_g):
    def func(k):
        return (
            -1
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 3
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (3 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 3
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (3 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            + (3 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (9 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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


def M_gg_0_2_3(sig_g):
    def func(k):
        return (
            -3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 1
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (39 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (15 / 16) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 195
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (315 / 16) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 315
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (315 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_0_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_2_4(sig_g):
    def func(k):
        return (
            -3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 1
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (39 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (15 / 16) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 195
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (315 / 16) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 315
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (315 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_0_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_0_2_5(sig_g):
    def func(k):
        return (
            (9 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 15
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 45
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (25 / 64) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 165
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (765 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (525 / 32) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (2625 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 1575
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (11025 / 64) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 11025
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (11025 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_0_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_0_4_0(sig_g):
    def func(k):
        return (
            (3 / 16)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 5
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 15
            / 8
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            - 105
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (105 / 16)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_0_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_0_4_1(sig_g):
    def func(k):
        return (
            (1 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            + (3 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (9 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_0_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_4_2(sig_g):
    def func(k):
        return (
            -3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 1
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (39 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (15 / 16) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 195
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (315 / 16) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 315
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (315 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_0_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_4_3(sig_g):
    def func(k):
        return (
            (3 / 16)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 5
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 15
            / 8
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            - 105
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (105 / 16)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def N_gg_0_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_0_4_4(sig_g):
    def func(k):
        return (
            -3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 1
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (39 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (15 / 16) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 195
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (315 / 16) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 315
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (315 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_0_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_4_5(sig_g):
    def func(k):
        return (
            (9 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 15
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 45
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (25 / 64) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 165
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (765 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (525 / 32) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (2625 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 1575
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (11025 / 64) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 11025
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (11025 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_0_4_5(theta, phi):
    return (
        (54675 / 18304) * np.cos(4 * phi)
        - 10935 / 11648 * np.cos(2 * theta)
        + (76545 / 36608) * np.cos(4 * theta)
        + (54675 / 18304) * np.cos(2 * phi - 3 * theta)
        - 10935 / 11648 * np.cos(2 * phi - theta)
        - 10935 / 11648 * np.cos(2 * phi + theta)
        + (54675 / 18304) * np.cos(2 * phi + 3 * theta)
        + (76545 / 36608) * np.cos(4 * phi - 2 * theta)
        + (76545 / 36608) * np.cos(4 * phi + 2 * theta)
        + 177147 / 256256
    )


def M_gg_0_6_0(sig_g):
    def func(k):
        return (
            -3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 1
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (39 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (15 / 16) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 195
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (315 / 16) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 315
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (315 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_0_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_0_6_1(sig_g):
    def func(k):
        return (
            -3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 1
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (39 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (15 / 16) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 195
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (315 / 16) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 315
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (315 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_0_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_0_6_2(sig_g):
    def func(k):
        return (
            (9 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
            - 15
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 45
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (25 / 64) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 165
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (765 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (525 / 32) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (2625 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 1575
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (11025 / 64) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 11025
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (11025 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_0_6_2(theta, phi):
    return (
        -2835 / 1408 * np.cos(4 * phi)
        + (225 / 128) * np.cos(2 * theta)
        + (1575 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(6 * phi - theta)
        + (945 / 256) * np.cos(6 * phi + theta)
        - 2025 / 2816
    )


def M_gg_1_0_0(sig_g):
    def func(k):
        return -np.sqrt(2) * np.sqrt(np.pi) * np.exp(
            -1 / 2 * k**2 * sig_g**2
        ) * erf((1 / 2) * np.sqrt(2) * k * sig_g) / (k**3 * sig_g**3) + np.pi * erf(
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (1 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + 3 * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (3 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3 * np.pi * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2 / (k**6 * sig_g**6)
            + (27 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 27
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (27 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_1_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_1_0_2(sig_g):
    def func(k):
        return (
            -3
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (9 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (5 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 32
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 45
            / 8
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (1265 / 32) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (135 / 32)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (2295 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (5775 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            + (6825 / 32)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1575
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (55125 / 32) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 55125
            / 32
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (55125 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_1_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_1_2_0(sig_g):
    def func(k):
        return (
            -1
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 1
            / 2
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (3 / 2) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + 3 * np.pi * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2 / (k**6 * sig_g**6)
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 1
            / 2
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (3 / 2) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + 3 * np.pi * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2 / (k**6 * sig_g**6)
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (1 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + 3 * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (3 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3 * np.pi * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2 / (k**6 * sig_g**6)
            + (27 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 27
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (27 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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


def M_gg_1_2_3(sig_g):
    def func(k):
        return (
            (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (17 / 8) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (39 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (255 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (165 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 585
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (315 / 2) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 315
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (315 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_1_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_2_4(sig_g):
    def func(k):
        return (
            (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (17 / 8) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (39 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (255 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (165 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 585
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (315 / 2) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 315
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (315 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_1_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_1_2_5(sig_g):
    def func(k):
        return (
            -3
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (9 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (5 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 32
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 45
            / 8
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (1265 / 32) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (135 / 32)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (2295 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (5775 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            + (6825 / 32)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1575
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (55125 / 32) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 55125
            / 32
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (55125 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_1_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_1_4_0(sig_g):
    def func(k):
        return (
            -11
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (3 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (5 / 8) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 15
            / 2
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (105 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 735
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (315 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_1_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_1_4_1(sig_g):
    def func(k):
        return (
            (1 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (1 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + 3 * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (3 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3 * np.pi * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2 / (k**6 * sig_g**6)
            + (27 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 27
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (27 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_1_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_4_2(sig_g):
    def func(k):
        return (
            (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (17 / 8) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (39 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (255 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (165 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 585
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (315 / 2) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 315
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (315 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_1_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_4_3(sig_g):
    def func(k):
        return (
            -11
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (3 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (5 / 8) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 15
            / 2
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (105 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 735
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (315 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
        )

    return func


def N_gg_1_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_1_4_4(sig_g):
    def func(k):
        return (
            (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (17 / 8) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (39 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (255 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (165 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 585
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (315 / 2) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 315
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (315 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_1_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_4_5(sig_g):
    def func(k):
        return (
            -3
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (9 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (5 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 32
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 45
            / 8
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (1265 / 32) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (135 / 32)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (2295 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (5775 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            + (6825 / 32)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1575
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (55125 / 32) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 55125
            / 32
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (55125 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_1_4_5(theta, phi):
    return (
        (54675 / 18304) * np.cos(4 * phi)
        - 10935 / 11648 * np.cos(2 * theta)
        + (76545 / 36608) * np.cos(4 * theta)
        + (54675 / 18304) * np.cos(2 * phi - 3 * theta)
        - 10935 / 11648 * np.cos(2 * phi - theta)
        - 10935 / 11648 * np.cos(2 * phi + theta)
        + (54675 / 18304) * np.cos(2 * phi + 3 * theta)
        + (76545 / 36608) * np.cos(4 * phi - 2 * theta)
        + (76545 / 36608) * np.cos(4 * phi + 2 * theta)
        + 177147 / 256256
    )


def M_gg_1_6_0(sig_g):
    def func(k):
        return (
            (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (17 / 8) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (39 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (255 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (165 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 585
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (315 / 2) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 315
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (315 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_1_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_1_6_1(sig_g):
    def func(k):
        return (
            (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            - 3
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (17 / 8) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (45 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (39 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (255 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (165 / 8)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 585
            / 16
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (315 / 2) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 315
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (315 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_1_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_1_6_2(sig_g):
    def func(k):
        return (
            -3
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**3 * sig_g**3)
            + (9 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**4 * sig_g**4)
            + (5 / 4) * np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 15
            / 32
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 45
            / 8
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (1265 / 32) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (135 / 32)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (2295 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (5775 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            + (6825 / 32)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1575
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (55125 / 32) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 55125
            / 32
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (55125 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_1_6_2(theta, phi):
    return (
        -2835 / 1408 * np.cos(4 * phi)
        + (225 / 128) * np.cos(2 * theta)
        + (1575 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(6 * phi - theta)
        + (945 / 256) * np.cos(6 * phi + theta)
        - 2025 / 2816
    )


def M_gg_2_0_0(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 2)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + 9 * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 9
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (81 / 4) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 81
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (81 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_2_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_0_2(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 3
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (85 / 4) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (465 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 135
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (15625 / 64) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            + (1875 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (5625 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (44625 / 32) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            + (2625 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            - 23625
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
            + (275625 / 64) * np.exp(-(k**2) * sig_g**2) / (k**12 * sig_g**12)
            - 275625
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**13 * sig_g**13)
            + (275625 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**14 * sig_g**14)
        )

    return func


def N_gg_2_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_2_2_0(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 1
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 1
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (9 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (9 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 1
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (9 / 2) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (9 / 4)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + 9 * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 9
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (81 / 4) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 81
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (81 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
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


def M_gg_2_2_3(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (121 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (83 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (117 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (1815 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 15
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1335
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (4725 / 16) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 4725
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (4725 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_2_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_2_4(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (121 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (83 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (117 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (1815 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 15
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1335
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (4725 / 16) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 4725
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (4725 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_2_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_2_2_5(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 3
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (85 / 4) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (465 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 135
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (15625 / 64) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            + (1875 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (5625 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (44625 / 32) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            + (2625 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            - 23625
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
            + (275625 / 64) * np.exp(-(k**2) * sig_g**2) / (k**12 * sig_g**12)
            - 275625
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**13 * sig_g**13)
            + (275625 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**14 * sig_g**14)
        )

    return func


def N_gg_2_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_2_4_0(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 11
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (3 / 16)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (85 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (5 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 45
            / 8
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (525 / 8) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 525
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (525 / 16)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_2_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_2_4_1(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 2)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + 9 * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            - 9
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 9
            / 4
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (81 / 4) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 81
            / 4
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (81 / 8)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_2_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_4_2(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (121 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (83 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (117 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (1815 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 15
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1335
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (4725 / 16) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 4725
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (4725 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_2_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_4_3(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 11
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (3 / 16)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (85 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (5 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 45
            / 8
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (525 / 8) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 525
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (525 / 16)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
        )

    return func


def N_gg_2_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_2_4_4(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (121 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (83 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (117 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (1815 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 15
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1335
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (4725 / 16) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 4725
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (4725 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_2_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_4_5(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 3
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (85 / 4) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (465 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 135
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (15625 / 64) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            + (1875 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (5625 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (44625 / 32) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            + (2625 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            - 23625
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
            + (275625 / 64) * np.exp(-(k**2) * sig_g**2) / (k**12 * sig_g**12)
            - 275625
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**13 * sig_g**13)
            + (275625 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**14 * sig_g**14)
        )

    return func


def N_gg_2_4_5(theta, phi):
    return (
        (54675 / 18304) * np.cos(4 * phi)
        - 10935 / 11648 * np.cos(2 * theta)
        + (76545 / 36608) * np.cos(4 * theta)
        + (54675 / 18304) * np.cos(2 * phi - 3 * theta)
        - 10935 / 11648 * np.cos(2 * phi - theta)
        - 10935 / 11648 * np.cos(2 * phi + theta)
        + (54675 / 18304) * np.cos(2 * phi + 3 * theta)
        + (76545 / 36608) * np.cos(4 * phi - 2 * theta)
        + (76545 / 36608) * np.cos(4 * phi + 2 * theta)
        + 177147 / 256256
    )


def M_gg_2_6_0(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (121 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (83 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (117 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (1815 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 15
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1335
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (4725 / 16) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 4725
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (4725 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_2_6_0(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + (4725 / 512) * np.cos(6 * phi - theta)
        + 3375 / 2816
    )


def M_gg_2_6_1(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            + (1 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            - 3
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (121 / 8) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (83 / 16)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            + (117 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (1815 / 16) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            - 15
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            - 1335
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (4725 / 16) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            - 4725
            / 16
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            + (4725 / 32)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
        )

    return func


def N_gg_2_6_1(theta, phi):
    return (
        (4725 / 1408) * np.cos(4 * phi)
        + (1125 / 1408) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(6 * phi + theta)
        + 3375 / 2816
    )


def M_gg_2_6_2(sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - 3
            / 8
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (9 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
            + (85 / 4) * np.exp(-(k**2) * sig_g**2) / (k**6 * sig_g**6)
            + (465 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**7 * sig_g**7)
            - 135
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**8 * sig_g**8)
            + (15625 / 64) * np.exp(-(k**2) * sig_g**2) / (k**8 * sig_g**8)
            + (1875 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**9 * sig_g**9)
            + (5625 / 64)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**10 * sig_g**10)
            + (44625 / 32) * np.exp(-(k**2) * sig_g**2) / (k**10 * sig_g**10)
            + (2625 / 64)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**11 * sig_g**11)
            - 23625
            / 32
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**12 * sig_g**12)
            + (275625 / 64) * np.exp(-(k**2) * sig_g**2) / (k**12 * sig_g**12)
            - 275625
            / 64
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**13 * sig_g**13)
            + (275625 / 128)
            * np.pi
            * erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**14 * sig_g**14)
        )

    return func


def N_gg_2_6_2(theta, phi):
    return (
        -2835 / 1408 * np.cos(4 * phi)
        + (225 / 128) * np.cos(2 * theta)
        + (1575 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(6 * phi - theta)
        + (945 / 256) * np.cos(6 * phi + theta)
        - 2025 / 2816
    )


def M_gv_0_1_0(sig_g):
    def func(k):
        return (
            (50 / 3)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**2 * sig_g)
        )

    return func


def N_gv_0_1_0(theta, phi):
    return -3 * np.cos(phi - 1 / 2 * theta)


def M_gv_0_1_1(sig_g):
    def func(k):
        return (
            -25
            / 3
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**2 * sig_g)
            - 50 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            + 25
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
        )

    return func


def N_gv_0_1_1(theta, phi):
    return -3 / 2 * np.cos(phi - 1 / 2 * theta) - 9 / 2 * np.cos(phi + (3 / 2) * theta)


def M_gv_0_3_0(sig_g):
    def func(k):
        return (
            -25
            / 3
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**2 * sig_g)
            - 50 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            + 25
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
        )

    return func


def N_gv_0_3_0(theta, phi):
    return (
        -9 / 4 * np.cos(phi - 1 / 2 * theta)
        - 9 / 8 * np.cos(phi + (3 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi + (1 / 2) * theta)
    )


def M_gv_0_3_1(sig_g):
    def func(k):
        return (
            (25 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**2 * sig_g)
            - 125 / 6 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 125
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 875 / 2 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            + (875 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def N_gv_0_3_1(theta, phi):
    return (
        -27 / 16 * np.cos(phi - 1 / 2 * theta)
        - 45 / 16 * np.cos(phi + (3 / 2) * theta)
        - 15 / 16 * np.cos(3 * phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi + (5 / 2) * theta)
    )


def M_gv_0_5_0(sig_g):
    def func(k):
        return (
            (25 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**2 * sig_g)
            - 125 / 6 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 125
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 875 / 2 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            + (875 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def N_gv_0_5_0(theta, phi):
    return (
        -135 / 64 * np.cos(phi - 1 / 2 * theta)
        - 45 / 32 * np.cos(phi + (3 / 2) * theta)
        - 105 / 32 * np.cos(3 * phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi + (5 / 2) * theta)
        - 945 / 128 * np.cos(5 * phi + (3 / 2) * theta)
    )


def M_gv_1_1_0(sig_g):
    def func(k):
        return -100 / 3 * np.exp(-1 / 2 * k**2 * sig_g**2) / (
            k**3 * sig_g**2
        ) + (50 / 3) * np.sqrt(2) * np.sqrt(np.pi) * erf(
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
            -100 / 3 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 25
            / 3
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 150 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            + 75
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def N_gv_1_1_1(theta, phi):
    return -3 / 2 * np.cos(phi - 1 / 2 * theta) - 9 / 2 * np.cos(phi + (3 / 2) * theta)


def M_gv_1_3_0(sig_g):
    def func(k):
        return (
            -100 / 3 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            - 25
            / 3
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 150 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            + 75
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
        )

    return func


def N_gv_1_3_0(theta, phi):
    return (
        -9 / 4 * np.cos(phi - 1 / 2 * theta)
        - 9 / 8 * np.cos(phi + (3 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi + (1 / 2) * theta)
    )


def M_gv_1_3_1(sig_g):
    def func(k):
        return (
            -100 / 3 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            + (25 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 2125 / 6 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            - 375
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
            - 4375 / 2 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**7 * sig_g**6)
            + (4375 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**8 * sig_g**7)
        )

    return func


def N_gv_1_3_1(theta, phi):
    return (
        -27 / 16 * np.cos(phi - 1 / 2 * theta)
        - 45 / 16 * np.cos(phi + (3 / 2) * theta)
        - 15 / 16 * np.cos(3 * phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi + (5 / 2) * theta)
    )


def M_gv_1_5_0(sig_g):
    def func(k):
        return (
            -100 / 3 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**3 * sig_g**2)
            + (25 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**4 * sig_g**3)
            - 2125 / 6 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**5 * sig_g**4)
            - 375
            / 2
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**6 * sig_g**5)
            - 4375 / 2 * np.exp(-1 / 2 * k**2 * sig_g**2) / (k**7 * sig_g**6)
            + (4375 / 4)
            * np.sqrt(2)
            * np.sqrt(np.pi)
            * erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**8 * sig_g**7)
        )

    return func


def N_gv_1_5_0(theta, phi):
    return (
        -135 / 64 * np.cos(phi - 1 / 2 * theta)
        - 45 / 32 * np.cos(phi + (3 / 2) * theta)
        - 105 / 32 * np.cos(3 * phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi + (5 / 2) * theta)
        - 945 / 128 * np.cos(5 * phi + (3 / 2) * theta)
    )


def M_vv_0_0_0(sig_g):
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_0_2_0(sig_g):
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


dictionary_terms = {"gg": ["0", "1", "2"], "gv": ["0", "1"], "vv": ["0"]}
dictionary_lmax = {"gg": [6, 6, 6], "gv": [5, 5], "vv": [2]}
dictionary_subterms = {
    "gg_0_0": 3,
    "gg_0_1": 0,
    "gg_0_2": 6,
    "gg_0_3": 0,
    "gg_0_4": 6,
    "gg_0_5": 0,
    "gg_0_6": 3,
    "gg_1_0": 3,
    "gg_1_1": 0,
    "gg_1_2": 6,
    "gg_1_3": 0,
    "gg_1_4": 6,
    "gg_1_5": 0,
    "gg_1_6": 3,
    "gg_2_0": 3,
    "gg_2_1": 0,
    "gg_2_2": 6,
    "gg_2_3": 0,
    "gg_2_4": 6,
    "gg_2_5": 0,
    "gg_2_6": 3,
    "gv_0_0": 0,
    "gv_0_1": 2,
    "gv_0_2": 0,
    "gv_0_3": 2,
    "gv_0_4": 0,
    "gv_0_5": 1,
    "gv_1_0": 0,
    "gv_1_1": 2,
    "gv_1_2": 0,
    "gv_1_3": 2,
    "gv_1_4": 0,
    "gv_1_5": 1,
    "vv_0_0": 1,
    "vv_0_1": 0,
    "vv_0_2": 1,
}
multi_index_model = False
