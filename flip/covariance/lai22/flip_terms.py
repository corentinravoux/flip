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



def M_gg_0_0_0_0():
    def func(k):
        return 1

    return func


def N_gg_0_0_0_0(theta, phi):
    return 1


def M_gg_0_1_0_0():
    def func(k):
        return -1 / 3 * k**2

    return func


def N_gg_0_1_0_0(theta, phi):
    return 1


def M_gg_0_1_2_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_0_1_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_0_1_2_1():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_0_1_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_0_2_0_0():
    def func(k):
        return (7 / 90) * k**4

    return func


def N_gg_0_2_0_0(theta, phi):
    return 1


def M_gg_0_2_0_1():
    def func(k):
        return (1 / 225) * k**4

    return func


def N_gg_0_2_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_0_2_2_0():
    def func(k):
        return (8 / 315) * k**4

    return func


def N_gg_0_2_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_0_2_2_1():
    def func(k):
        return (8 / 315) * k**4

    return func


def N_gg_0_2_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_0_2_2_2():
    def func(k):
        return (1 / 225) * k**4

    return func


def N_gg_0_2_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_0_2_4_0():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_0_2_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_0_2_4_1():
    def func(k):
        return (1 / 225) * k**4

    return func


def N_gg_0_2_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_2_4_2():
    def func(k):
        return (1 / 315) * k**4

    return func


def N_gg_0_2_4_2(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_0_3_0_0():
    def func(k):
        return -1 / 70 * k**6

    return func


def N_gg_0_3_0_0(theta, phi):
    return 1


def M_gg_0_3_0_1():
    def func(k):
        return -1 / 525 * k**6

    return func


def N_gg_0_3_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_0_3_2_0():
    def func(k):
        return -19 / 3150 * k**6

    return func


def N_gg_0_3_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_0_3_2_1():
    def func(k):
        return -19 / 3150 * k**6

    return func


def N_gg_0_3_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_0_3_2_2():
    def func(k):
        return -1 / 525 * k**6

    return func


def N_gg_0_3_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_0_3_2_3():
    def func(k):
        return -1 / 4725 * k**6

    return func


def N_gg_0_3_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_0_3_2_4():
    def func(k):
        return -1 / 4725 * k**6

    return func


def N_gg_0_3_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_3_4_0():
    def func(k):
        return -13 / 10395 * k**6

    return func


def N_gg_0_3_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_0_3_4_1():
    def func(k):
        return -1 / 525 * k**6

    return func


def N_gg_0_3_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_3_4_2():
    def func(k):
        return -1 / 4725 * k**6

    return func


def N_gg_0_3_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_3_4_3():
    def func(k):
        return -13 / 10395 * k**6

    return func


def N_gg_0_3_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_0_3_4_4():
    def func(k):
        return -1 / 4725 * k**6

    return func


def N_gg_0_3_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_3_6_0():
    def func(k):
        return -1 / 9009 * k**6

    return func


def N_gg_0_3_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_0_3_6_1():
    def func(k):
        return -1 / 4725 * k**6

    return func


def N_gg_0_3_6_1(theta, phi):
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


def M_gg_0_3_6_2():
    def func(k):
        return -1 / 4725 * k**6

    return func


def N_gg_0_3_6_2(theta, phi):
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


def M_gg_0_3_6_3():
    def func(k):
        return -1 / 9009 * k**6

    return func


def N_gg_0_3_6_3(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_0_4_0_0():
    def func(k):
        return (163 / 100800) * k**8

    return func


def N_gg_0_4_0_0(theta, phi):
    return 1


def M_gg_0_4_0_1():
    def func(k):
        return (31 / 66150) * k**8

    return func


def N_gg_0_4_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_0_4_0_2():
    def func(k):
        return (1 / 99225) * k**8

    return func


def N_gg_0_4_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_0_4_2_0():
    def func(k):
        return (67 / 75600) * k**8

    return func


def N_gg_0_4_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_0_4_2_1():
    def func(k):
        return (67 / 75600) * k**8

    return func


def N_gg_0_4_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_0_4_2_2():
    def func(k):
        return (31 / 66150) * k**8

    return func


def N_gg_0_4_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_0_4_2_3():
    def func(k):
        return (34 / 363825) * k**8

    return func


def N_gg_0_4_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_0_4_2_4():
    def func(k):
        return (34 / 363825) * k**8

    return func


def N_gg_0_4_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_4_2_5():
    def func(k):
        return (1 / 99225) * k**8

    return func


def N_gg_0_4_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_0_4_4_0():
    def func(k):
        return (83 / 415800) * k**8

    return func


def N_gg_0_4_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_0_4_4_1():
    def func(k):
        return (31 / 66150) * k**8

    return func


def N_gg_0_4_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_4_4_2():
    def func(k):
        return (34 / 363825) * k**8

    return func


def N_gg_0_4_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_4_4_3():
    def func(k):
        return (1 / 135135) * k**8

    return func


def N_gg_0_4_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_0_4_4_4():
    def func(k):
        return (83 / 415800) * k**8

    return func


def N_gg_0_4_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_0_4_4_5():
    def func(k):
        return (34 / 363825) * k**8

    return func


def N_gg_0_4_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_4_4_6():
    def func(k):
        return (1 / 99225) * k**8

    return func


def N_gg_0_4_4_6(theta, phi):
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


def M_gg_0_4_4_7():
    def func(k):
        return (1 / 135135) * k**8

    return func


def N_gg_0_4_4_7(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_0_4_6_0():
    def func(k):
        return (1 / 54054) * k**8

    return func


def N_gg_0_4_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_0_4_6_1():
    def func(k):
        return (34 / 363825) * k**8

    return func


def N_gg_0_4_6_1(theta, phi):
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


def M_gg_0_4_6_2():
    def func(k):
        return (1 / 135135) * k**8

    return func


def N_gg_0_4_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_0_4_6_3():
    def func(k):
        return (34 / 363825) * k**8

    return func


def N_gg_0_4_6_3(theta, phi):
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


def M_gg_0_4_6_4():
    def func(k):
        return (1 / 99225) * k**8

    return func


def N_gg_0_4_6_4(theta, phi):
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


def M_gg_0_4_6_5():
    def func(k):
        return (1 / 54054) * k**8

    return func


def N_gg_0_4_6_5(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_0_4_6_6():
    def func(k):
        return (1 / 135135) * k**8

    return func


def N_gg_0_4_6_6(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_0_4_8_0():
    def func(k):
        return (1 / 135135) * k**8

    return func


def N_gg_0_4_8_0(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_0_4_8_1():
    def func(k):
        return (1 / 99225) * k**8

    return func


def N_gg_0_4_8_1(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_0_4_8_2():
    def func(k):
        return (1 / 135135) * k**8

    return func


def N_gg_0_4_8_2(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_0_5_0_0():
    def func(k):
        return -1 / 6720 * k**10

    return func


def N_gg_0_5_0_0(theta, phi):
    return 1


def M_gg_0_5_0_1():
    def func(k):
        return -1 / 17640 * k**10

    return func


def N_gg_0_5_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_0_5_0_2():
    def func(k):
        return -1 / 218295 * k**10

    return func


def N_gg_0_5_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_0_5_2_0():
    def func(k):
        return -13 / 141120 * k**10

    return func


def N_gg_0_5_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_0_5_2_1():
    def func(k):
        return -13 / 141120 * k**10

    return func


def N_gg_0_5_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_0_5_2_2():
    def func(k):
        return -1 / 17640 * k**10

    return func


def N_gg_0_5_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_0_5_2_3():
    def func(k):
        return -29 / 1746360 * k**10

    return func


def N_gg_0_5_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_0_5_2_4():
    def func(k):
        return -29 / 1746360 * k**10

    return func


def N_gg_0_5_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_5_2_5():
    def func(k):
        return -1 / 218295 * k**10

    return func


def N_gg_0_5_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_0_5_2_6():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_0_5_2_7():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_0_5_4_0():
    def func(k):
        return -2 / 72765 * k**10

    return func


def N_gg_0_5_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_0_5_4_1():
    def func(k):
        return -1 / 17640 * k**10

    return func


def N_gg_0_5_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_5_4_2():
    def func(k):
        return -29 / 1746360 * k**10

    return func


def N_gg_0_5_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_5_4_3():
    def func(k):
        return -1 / 630630 * k**10

    return func


def N_gg_0_5_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_0_5_4_4():
    def func(k):
        return -2 / 72765 * k**10

    return func


def N_gg_0_5_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_0_5_4_5():
    def func(k):
        return -29 / 1746360 * k**10

    return func


def N_gg_0_5_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_5_4_6():
    def func(k):
        return -1 / 218295 * k**10

    return func


def N_gg_0_5_4_6(theta, phi):
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


def M_gg_0_5_4_7():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_0_5_4_8():
    def func(k):
        return -1 / 630630 * k**10

    return func


def N_gg_0_5_4_8(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_0_5_4_9():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_4_9(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_0_5_6_0():
    def func(k):
        return -1 / 360360 * k**10

    return func


def N_gg_0_5_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_0_5_6_1():
    def func(k):
        return -29 / 1746360 * k**10

    return func


def N_gg_0_5_6_1(theta, phi):
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


def M_gg_0_5_6_2():
    def func(k):
        return -1 / 630630 * k**10

    return func


def N_gg_0_5_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_0_5_6_3():
    def func(k):
        return -29 / 1746360 * k**10

    return func


def N_gg_0_5_6_3(theta, phi):
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


def M_gg_0_5_6_4():
    def func(k):
        return -1 / 218295 * k**10

    return func


def N_gg_0_5_6_4(theta, phi):
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


def M_gg_0_5_6_5():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_6_5(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_0_5_6_6():
    def func(k):
        return -1 / 360360 * k**10

    return func


def N_gg_0_5_6_6(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_0_5_6_7():
    def func(k):
        return -1 / 630630 * k**10

    return func


def N_gg_0_5_6_7(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_0_5_6_8():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_6_8(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_0_5_8_0():
    def func(k):
        return -1 / 630630 * k**10

    return func


def N_gg_0_5_8_0(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_0_5_8_1():
    def func(k):
        return -1 / 218295 * k**10

    return func


def N_gg_0_5_8_1(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_0_5_8_2():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_8_2(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_0_5_8_3():
    def func(k):
        return -1 / 630630 * k**10

    return func


def N_gg_0_5_8_3(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_0_5_8_4():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_0_5_10_0():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_10_0(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_0_5_10_1():
    def func(k):
        return -1 / 2837835 * k**10

    return func


def N_gg_0_5_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_0_6_0_0():
    def func(k):
        return (1 / 112896) * k**12

    return func


def N_gg_0_6_0_0(theta, phi):
    return 1


def M_gg_0_6_0_1():
    def func(k):
        return (1 / 254016) * k**12

    return func


def N_gg_0_6_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_0_6_0_2():
    def func(k):
        return (1 / 1920996) * k**12

    return func


def N_gg_0_6_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_0_6_0_3():
    def func(k):
        return (1 / 81162081) * k**12

    return func


def N_gg_0_6_0_3(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * theta)
        + (819 / 256) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + 325 / 256
    )


def M_gg_0_6_2_0():
    def func(k):
        return (1 / 169344) * k**12

    return func


def N_gg_0_6_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_0_6_2_1():
    def func(k):
        return (1 / 169344) * k**12

    return func


def N_gg_0_6_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_0_6_2_2():
    def func(k):
        return (1 / 254016) * k**12

    return func


def N_gg_0_6_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_0_6_2_3():
    def func(k):
        return (1 / 698544) * k**12

    return func


def N_gg_0_6_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_0_6_2_4():
    def func(k):
        return (1 / 698544) * k**12

    return func


def N_gg_0_6_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_6_2_5():
    def func(k):
        return (1 / 1920996) * k**12

    return func


def N_gg_0_6_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_0_6_2_6():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_0_6_2_7():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_0_6_2_8():
    def func(k):
        return (1 / 81162081) * k**12

    return func


def N_gg_0_6_2_8(theta, phi):
    return (
        -6825 / 5632 * np.cos(2 * theta)
        + (819 / 2816) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + (819 / 512) * np.cos(2 * phi - 5 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 512) * np.cos(2 * phi + 5 * theta)
        - 2275 / 2816
    )


def M_gg_0_6_4_0():
    def func(k):
        return (1 / 465696) * k**12

    return func


def N_gg_0_6_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_0_6_4_1():
    def func(k):
        return (1 / 254016) * k**12

    return func


def N_gg_0_6_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_0_6_4_2():
    def func(k):
        return (1 / 698544) * k**12

    return func


def N_gg_0_6_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_6_4_3():
    def func(k):
        return (1 / 4540536) * k**12

    return func


def N_gg_0_6_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_0_6_4_4():
    def func(k):
        return (1 / 465696) * k**12

    return func


def N_gg_0_6_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_0_6_4_5():
    def func(k):
        return (1 / 698544) * k**12

    return func


def N_gg_0_6_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_0_6_4_6():
    def func(k):
        return (1 / 1920996) * k**12

    return func


def N_gg_0_6_4_6(theta, phi):
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


def M_gg_0_6_4_7():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_0_6_4_8():
    def func(k):
        return (1 / 4540536) * k**12

    return func


def N_gg_0_6_4_8(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_0_6_4_9():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_4_9(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_0_6_4_10():
    def func(k):
        return (1 / 81162081) * k**12

    return func


def N_gg_0_6_4_10(theta, phi):
    return (
        (1003275 / 382976) * np.cos(4 * phi)
        + (12285 / 69632) * np.cos(2 * theta)
        - 22113 / 11968 * np.cos(4 * theta)
        + (243243 / 69632) * np.cos(6 * theta)
        + (110565 / 34816) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(2 * phi + 5 * theta)
        + (85995 / 69632) * np.cos(4 * phi - 4 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (85995 / 69632) * np.cos(4 * phi + 4 * theta)
        + 61425 / 95744
    )


def M_gg_0_6_6_0():
    def func(k):
        return (1 / 3027024) * k**12

    return func


def N_gg_0_6_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_0_6_6_1():
    def func(k):
        return (1 / 698544) * k**12

    return func


def N_gg_0_6_6_1(theta, phi):
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


def M_gg_0_6_6_2():
    def func(k):
        return (1 / 4540536) * k**12

    return func


def N_gg_0_6_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_0_6_6_3():
    def func(k):
        return (1 / 698544) * k**12

    return func


def N_gg_0_6_6_3(theta, phi):
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


def M_gg_0_6_6_4():
    def func(k):
        return (1 / 1920996) * k**12

    return func


def N_gg_0_6_6_4(theta, phi):
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


def M_gg_0_6_6_5():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_6_5(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_0_6_6_6():
    def func(k):
        return (1 / 3027024) * k**12

    return func


def N_gg_0_6_6_6(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_0_6_6_7():
    def func(k):
        return (1 / 4540536) * k**12

    return func


def N_gg_0_6_6_7(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_0_6_6_8():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_6_8(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_0_6_6_9():
    def func(k):
        return (1 / 81162081) * k**12

    return func


def N_gg_0_6_6_9(theta, phi):
    return (
        -5589675 / 3638272 * np.cos(4 * phi)
        + (443625 / 661504) * np.cos(2 * theta)
        + (266175 / 909568) * np.cos(4 * theta)
        + (975975 / 661504) * np.cos(6 * theta)
        + (1863225 / 661504) * np.cos(2 * phi - 5 * theta)
        - 5589675 / 3638272 * np.cos(2 * phi - 3 * theta)
        + (443625 / 661504) * np.cos(2 * phi - theta)
        + (443625 / 661504) * np.cos(2 * phi + theta)
        - 5589675 / 3638272 * np.cos(2 * phi + 3 * theta)
        + (1863225 / 661504) * np.cos(2 * phi + 5 * theta)
        + (1863225 / 661504) * np.cos(4 * phi - 4 * theta)
        + (266175 / 909568) * np.cos(4 * phi - 2 * theta)
        + (266175 / 909568) * np.cos(4 * phi + 2 * theta)
        + (1863225 / 661504) * np.cos(4 * phi + 4 * theta)
        + (975975 / 661504) * np.cos(6 * phi - 3 * theta)
        + (1863225 / 661504) * np.cos(6 * phi - theta)
        + (1863225 / 661504) * np.cos(6 * phi + theta)
        + (975975 / 661504) * np.cos(6 * phi + 3 * theta)
        - 528125 / 909568
    )


def M_gg_0_6_8_0():
    def func(k):
        return (1 / 4540536) * k**12

    return func


def N_gg_0_6_8_0(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_0_6_8_1():
    def func(k):
        return (1 / 1920996) * k**12

    return func


def N_gg_0_6_8_1(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_0_6_8_2():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_8_2(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_0_6_8_3():
    def func(k):
        return (1 / 4540536) * k**12

    return func


def N_gg_0_6_8_3(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_0_6_8_4():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_0_6_8_5():
    def func(k):
        return (1 / 81162081) * k**12

    return func


def N_gg_0_6_8_5(theta, phi):
    return (
        (429975 / 311296) * np.cos(4 * phi)
        + (2395575 / 622592) * np.cos(8 * phi)
        - 16960125 / 13697024 * np.cos(2 * theta)
        + (12755925 / 6848512) * np.cos(4 * theta)
        + (525525 / 1245184) * np.cos(6 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (429975 / 311296) * np.cos(2 * phi + 5 * theta)
        + (1576575 / 622592) * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        + (1576575 / 622592) * np.cos(4 * phi + 4 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        - 266175 / 311296 * np.cos(6 * phi - theta)
        - 266175 / 311296 * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi - 2 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi + 2 * theta)
        + 1990625 / 3424256
    )


def M_gg_0_6_10_0():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_10_0(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_0_6_10_1():
    def func(k):
        return (1 / 12486474) * k**12

    return func


def N_gg_0_6_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_0_6_10_2():
    def func(k):
        return (1 / 81162081) * k**12

    return func


def N_gg_0_6_10_2(theta, phi):
    return (
        -184459275 / 121716736 * np.cos(4 * phi)
        - 36891855 / 14319616 * np.cos(8 * phi)
        + (568856925 / 486866944) * np.cos(2 * theta)
        + (269800713 / 243433472) * np.cos(4 * theta)
        + (35756721 / 486866944) * np.cos(6 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi - 5 * theta)
        + (854188335 / 486866944) * np.cos(2 * phi - 3 * theta)
        - 99324225 / 243433472 * np.cos(2 * phi - theta)
        - 99324225 / 243433472 * np.cos(2 * phi + theta)
        + (854188335 / 486866944) * np.cos(2 * phi + 3 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi + 5 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi + 2 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi + 4 * theta)
        + (553377825 / 243433472) * np.cos(6 * phi - 3 * theta)
        - 110675565 / 486866944 * np.cos(6 * phi - theta)
        - 110675565 / 486866944 * np.cos(6 * phi + theta)
        + (553377825 / 243433472) * np.cos(6 * phi + 3 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi - 2 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi + 2 * theta)
        + (7378371 / 1507328) * np.cos(10 * phi - theta)
        + (7378371 / 1507328) * np.cos(10 * phi + theta)
        - 81265275 / 121716736
    )


def M_gg_0_6_12_0():
    def func(k):
        return (1 / 81162081) * k**12

    return func


def N_gg_0_6_12_0(theta, phi):
    return (
        (10145260125 / 3894935552) * np.cos(4 * phi)
        + (11594583 / 3014656) * np.cos(8 * phi)
        + (9018009 / 524288) * np.cos(12 * phi)
        + (5150670525 / 3894935552) * np.cos(2 * theta)
        + (206026821 / 973733888) * np.cos(4 * theta)
        + (22891869 / 3894935552) * np.cos(6 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi - 5 * theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi - 3 * theta)
        + (2029052025 / 973733888) * np.cos(2 * phi - theta)
        + (2029052025 / 973733888) * np.cos(2 * phi + theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi + 3 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi + 5 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi - 4 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi - 2 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi + 2 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi + 4 * theta)
        + (32207175 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (289864575 / 114556928) * np.cos(6 * phi - theta)
        + (289864575 / 114556928) * np.cos(6 * phi + theta)
        + (32207175 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (27054027 / 6029312) * np.cos(10 * phi - theta)
        + (27054027 / 6029312) * np.cos(10 * phi + theta)
        + 572296725 / 486866944
    )


def M_gg_1_0_0_0():
    def func(k):
        return 2 / 3

    return func


def N_gg_1_0_0_0(theta, phi):
    return 1


def M_gg_1_0_2_0():
    def func(k):
        return 2 / 15

    return func


def N_gg_1_0_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_1_0_2_1():
    def func(k):
        return 2 / 15

    return func


def N_gg_1_0_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_1_1_0_0():
    def func(k):
        return -14 / 45 * k**2

    return func


def N_gg_1_1_0_0(theta, phi):
    return 1


def M_gg_1_1_0_1():
    def func(k):
        return -4 / 225 * k**2

    return func


def N_gg_1_1_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_1_1_2_0():
    def func(k):
        return -32 / 315 * k**2

    return func


def N_gg_1_1_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_1_1_2_1():
    def func(k):
        return -32 / 315 * k**2

    return func


def N_gg_1_1_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_1_1_2_2():
    def func(k):
        return -4 / 225 * k**2

    return func


def N_gg_1_1_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_1_1_4_0():
    def func(k):
        return -4 / 315 * k**2

    return func


def N_gg_1_1_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_1_1_4_1():
    def func(k):
        return -4 / 225 * k**2

    return func


def N_gg_1_1_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_1_4_2():
    def func(k):
        return -4 / 315 * k**2

    return func


def N_gg_1_1_4_2(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_1_2_0_0():
    def func(k):
        return (3 / 35) * k**4

    return func


def N_gg_1_2_0_0(theta, phi):
    return 1


def M_gg_1_2_0_1():
    def func(k):
        return (2 / 175) * k**4

    return func


def N_gg_1_2_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_1_2_2_0():
    def func(k):
        return (19 / 525) * k**4

    return func


def N_gg_1_2_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_1_2_2_1():
    def func(k):
        return (19 / 525) * k**4

    return func


def N_gg_1_2_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_1_2_2_2():
    def func(k):
        return (2 / 175) * k**4

    return func


def N_gg_1_2_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_1_2_2_3():
    def func(k):
        return (2 / 1575) * k**4

    return func


def N_gg_1_2_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_1_2_2_4():
    def func(k):
        return (2 / 1575) * k**4

    return func


def N_gg_1_2_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_2_4_0():
    def func(k):
        return (26 / 3465) * k**4

    return func


def N_gg_1_2_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_1_2_4_1():
    def func(k):
        return (2 / 175) * k**4

    return func


def N_gg_1_2_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_2_4_2():
    def func(k):
        return (2 / 1575) * k**4

    return func


def N_gg_1_2_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_2_4_3():
    def func(k):
        return (26 / 3465) * k**4

    return func


def N_gg_1_2_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_1_2_4_4():
    def func(k):
        return (2 / 1575) * k**4

    return func


def N_gg_1_2_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_2_6_0():
    def func(k):
        return (2 / 3003) * k**4

    return func


def N_gg_1_2_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_1_2_6_1():
    def func(k):
        return (2 / 1575) * k**4

    return func


def N_gg_1_2_6_1(theta, phi):
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


def M_gg_1_2_6_2():
    def func(k):
        return (2 / 1575) * k**4

    return func


def N_gg_1_2_6_2(theta, phi):
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


def M_gg_1_2_6_3():
    def func(k):
        return (2 / 3003) * k**4

    return func


def N_gg_1_2_6_3(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_1_3_0_0():
    def func(k):
        return -83 / 4725 * k**6

    return func


def N_gg_1_3_0_0(theta, phi):
    return 1


def M_gg_1_3_0_1():
    def func(k):
        return -124 / 33075 * k**6

    return func


def N_gg_1_3_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_1_3_0_2():
    def func(k):
        return -8 / 99225 * k**6

    return func


def N_gg_1_3_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_1_3_2_0():
    def func(k):
        return -152 / 17325 * k**6

    return func


def N_gg_1_3_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_1_3_2_1():
    def func(k):
        return -152 / 17325 * k**6

    return func


def N_gg_1_3_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_1_3_2_2():
    def func(k):
        return -124 / 33075 * k**6

    return func


def N_gg_1_3_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_1_3_2_3():
    def func(k):
        return -272 / 363825 * k**6

    return func


def N_gg_1_3_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_1_3_2_4():
    def func(k):
        return -272 / 363825 * k**6

    return func


def N_gg_1_3_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_3_2_5():
    def func(k):
        return -8 / 99225 * k**6

    return func


def N_gg_1_3_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_1_3_4_0():
    def func(k):
        return -1604 / 675675 * k**6

    return func


def N_gg_1_3_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_1_3_4_1():
    def func(k):
        return -124 / 33075 * k**6

    return func


def N_gg_1_3_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_3_4_2():
    def func(k):
        return -272 / 363825 * k**6

    return func


def N_gg_1_3_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_3_4_3():
    def func(k):
        return -8 / 135135 * k**6

    return func


def N_gg_1_3_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_1_3_4_4():
    def func(k):
        return -1604 / 675675 * k**6

    return func


def N_gg_1_3_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_1_3_4_5():
    def func(k):
        return -272 / 363825 * k**6

    return func


def N_gg_1_3_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_3_4_6():
    def func(k):
        return -8 / 99225 * k**6

    return func


def N_gg_1_3_4_6(theta, phi):
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


def M_gg_1_3_4_7():
    def func(k):
        return -8 / 135135 * k**6

    return func


def N_gg_1_3_4_7(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_1_3_6_0():
    def func(k):
        return -16 / 45045 * k**6

    return func


def N_gg_1_3_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_1_3_6_1():
    def func(k):
        return -272 / 363825 * k**6

    return func


def N_gg_1_3_6_1(theta, phi):
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


def M_gg_1_3_6_2():
    def func(k):
        return -8 / 135135 * k**6

    return func


def N_gg_1_3_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_1_3_6_3():
    def func(k):
        return -272 / 363825 * k**6

    return func


def N_gg_1_3_6_3(theta, phi):
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


def M_gg_1_3_6_4():
    def func(k):
        return -8 / 99225 * k**6

    return func


def N_gg_1_3_6_4(theta, phi):
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


def M_gg_1_3_6_5():
    def func(k):
        return -16 / 45045 * k**6

    return func


def N_gg_1_3_6_5(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_1_3_6_6():
    def func(k):
        return -8 / 135135 * k**6

    return func


def N_gg_1_3_6_6(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_1_3_8_0():
    def func(k):
        return -8 / 328185 * k**6

    return func


def N_gg_1_3_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi + theta)
        + (11781 / 4096) * np.cos(4 * phi + 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi + 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi + 4 * theta)
        + 20825 / 16384
    )


def M_gg_1_3_8_1():
    def func(k):
        return -8 / 135135 * k**6

    return func


def N_gg_1_3_8_1(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_1_3_8_2():
    def func(k):
        return -8 / 99225 * k**6

    return func


def N_gg_1_3_8_2(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_1_3_8_3():
    def func(k):
        return -8 / 135135 * k**6

    return func


def N_gg_1_3_8_3(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_1_3_8_4():
    def func(k):
        return -8 / 328185 * k**6

    return func


def N_gg_1_3_8_4(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi - theta)
        + (11781 / 4096) * np.cos(4 * phi - 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi - 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi - 4 * theta)
        + 20825 / 16384
    )


def M_gg_1_4_0_0():
    def func(k):
        return (41 / 18144) * k**8

    return func


def N_gg_1_4_0_0(theta, phi):
    return 1


def M_gg_1_4_0_1():
    def func(k):
        return (691 / 873180) * k**8

    return func


def N_gg_1_4_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_1_4_0_2():
    def func(k):
        return (2 / 43659) * k**8

    return func


def N_gg_1_4_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_1_4_2_0():
    def func(k):
        return (9473 / 6985440) * k**8

    return func


def N_gg_1_4_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_1_4_2_1():
    def func(k):
        return (9473 / 6985440) * k**8

    return func


def N_gg_1_4_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_1_4_2_2():
    def func(k):
        return (691 / 873180) * k**8

    return func


def N_gg_1_4_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_1_4_2_3():
    def func(k):
        return (2473 / 11351340) * k**8

    return func


def N_gg_1_4_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_1_4_2_4():
    def func(k):
        return (2473 / 11351340) * k**8

    return func


def N_gg_1_4_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_4_2_5():
    def func(k):
        return (2 / 43659) * k**8

    return func


def N_gg_1_4_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_1_4_2_6():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_1_4_2_7():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_1_4_4_0():
    def func(k):
        return (17 / 42042) * k**8

    return func


def N_gg_1_4_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_1_4_4_1():
    def func(k):
        return (691 / 873180) * k**8

    return func


def N_gg_1_4_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_4_4_2():
    def func(k):
        return (2473 / 11351340) * k**8

    return func


def N_gg_1_4_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_4_4_3():
    def func(k):
        return (421 / 14189175) * k**8

    return func


def N_gg_1_4_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_1_4_4_4():
    def func(k):
        return (17 / 42042) * k**8

    return func


def N_gg_1_4_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_1_4_4_5():
    def func(k):
        return (2473 / 11351340) * k**8

    return func


def N_gg_1_4_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_4_4_6():
    def func(k):
        return (2 / 43659) * k**8

    return func


def N_gg_1_4_4_6(theta, phi):
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


def M_gg_1_4_4_7():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_1_4_4_8():
    def func(k):
        return (421 / 14189175) * k**8

    return func


def N_gg_1_4_4_8(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_1_4_4_9():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_4_9(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_1_4_6_0():
    def func(k):
        return (101 / 1621620) * k**8

    return func


def N_gg_1_4_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_1_4_6_1():
    def func(k):
        return (2473 / 11351340) * k**8

    return func


def N_gg_1_4_6_1(theta, phi):
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


def M_gg_1_4_6_2():
    def func(k):
        return (421 / 14189175) * k**8

    return func


def N_gg_1_4_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_1_4_6_3():
    def func(k):
        return (8 / 4922775) * k**8

    return func


def N_gg_1_4_6_3(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (9009 / 2048) * np.cos(4 * phi + 4 * theta)
        + (693 / 4096) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(6 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_1_4_6_4():
    def func(k):
        return (2473 / 11351340) * k**8

    return func


def N_gg_1_4_6_4(theta, phi):
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


def M_gg_1_4_6_5():
    def func(k):
        return (2 / 43659) * k**8

    return func


def N_gg_1_4_6_5(theta, phi):
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


def M_gg_1_4_6_6():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_6_6(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_1_4_6_7():
    def func(k):
        return (101 / 1621620) * k**8

    return func


def N_gg_1_4_6_7(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_1_4_6_8():
    def func(k):
        return (421 / 14189175) * k**8

    return func


def N_gg_1_4_6_8(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_1_4_6_9():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_6_9(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_1_4_6_10():
    def func(k):
        return (8 / 4922775) * k**8

    return func


def N_gg_1_4_6_10(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (9009 / 2048) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (45045 / 4096) * np.cos(6 * phi - 5 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (693 / 4096) * np.cos(6 * phi - theta)
        + 1225 / 1024
    )


def M_gg_1_4_8_0():
    def func(k):
        return (4 / 984555) * k**8

    return func


def N_gg_1_4_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi + theta)
        + (11781 / 4096) * np.cos(4 * phi + 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi + 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi + 4 * theta)
        + 20825 / 16384
    )


def M_gg_1_4_8_1():
    def func(k):
        return (421 / 14189175) * k**8

    return func


def N_gg_1_4_8_1(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_1_4_8_2():
    def func(k):
        return (8 / 4922775) * k**8

    return func


def N_gg_1_4_8_2(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (187425 / 77824) * np.cos(2 * phi - theta)
        - 26775 / 19456 * np.cos(2 * phi + theta)
        + (176715 / 77824) * np.cos(2 * phi + 3 * theta)
        - 11781 / 19456 * np.cos(4 * phi + 2 * theta)
        + (153153 / 77824) * np.cos(4 * phi + 4 * theta)
        + (153153 / 77824) * np.cos(6 * phi + theta)
        + (21879 / 19456) * np.cos(6 * phi + 3 * theta)
        + (109395 / 77824) * np.cos(6 * phi + 5 * theta)
        + (109395 / 77824) * np.cos(8 * phi + 2 * theta)
        + (546975 / 77824) * np.cos(8 * phi + 4 * theta)
        - 62475 / 77824
    )


def M_gg_1_4_8_3():
    def func(k):
        return (2 / 43659) * k**8

    return func


def N_gg_1_4_8_3(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_1_4_8_4():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_1_4_8_5():
    def func(k):
        return (421 / 14189175) * k**8

    return func


def N_gg_1_4_8_5(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_1_4_8_6():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_8_6(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_1_4_8_7():
    def func(k):
        return (4 / 984555) * k**8

    return func


def N_gg_1_4_8_7(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi - theta)
        + (11781 / 4096) * np.cos(4 * phi - 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi - 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi - 4 * theta)
        + 20825 / 16384
    )


def M_gg_1_4_8_8():
    def func(k):
        return (8 / 4922775) * k**8

    return func


def N_gg_1_4_8_8(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (176715 / 77824) * np.cos(2 * phi - 3 * theta)
        - 26775 / 19456 * np.cos(2 * phi - theta)
        + (187425 / 77824) * np.cos(2 * phi + theta)
        + (153153 / 77824) * np.cos(4 * phi - 4 * theta)
        - 11781 / 19456 * np.cos(4 * phi - 2 * theta)
        + (109395 / 77824) * np.cos(6 * phi - 5 * theta)
        + (21879 / 19456) * np.cos(6 * phi - 3 * theta)
        + (153153 / 77824) * np.cos(6 * phi - theta)
        + (546975 / 77824) * np.cos(8 * phi - 4 * theta)
        + (109395 / 77824) * np.cos(8 * phi - 2 * theta)
        - 62475 / 77824
    )


def M_gg_1_4_10_0():
    def func(k):
        return (8 / 4922775) * k**8

    return func


def N_gg_1_4_10_0(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (1819125 / 1245184) * np.cos(2 * phi - theta)
        + (363825 / 155648) * np.cos(2 * phi + theta)
        + (363825 / 622592) * np.cos(2 * phi + 3 * theta)
        + (675675 / 311296) * np.cos(4 * phi + 2 * theta)
        + (96525 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 622592) * np.cos(6 * phi + theta)
        + (289575 / 155648) * np.cos(6 * phi + 3 * theta)
        + (289575 / 2490368) * np.cos(6 * phi + 5 * theta)
        + (1640925 / 311296) * np.cos(8 * phi + 2 * theta)
        + (1640925 / 1245184) * np.cos(8 * phi + 4 * theta)
        + (1640925 / 131072) * np.cos(10 * phi + 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_1_4_10_1():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_1_4_10_2():
    def func(k):
        return (2 / 567567) * k**8

    return func


def N_gg_1_4_10_2(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_1_4_10_3():
    def func(k):
        return (8 / 4922775) * k**8

    return func


def N_gg_1_4_10_3(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (363825 / 622592) * np.cos(2 * phi - 3 * theta)
        + (363825 / 155648) * np.cos(2 * phi - theta)
        + (1819125 / 1245184) * np.cos(2 * phi + theta)
        + (96525 / 311296) * np.cos(4 * phi - 4 * theta)
        + (675675 / 311296) * np.cos(4 * phi - 2 * theta)
        + (289575 / 2490368) * np.cos(6 * phi - 5 * theta)
        + (289575 / 155648) * np.cos(6 * phi - 3 * theta)
        + (2027025 / 622592) * np.cos(6 * phi - theta)
        + (1640925 / 1245184) * np.cos(8 * phi - 4 * theta)
        + (1640925 / 311296) * np.cos(8 * phi - 2 * theta)
        + (1640925 / 131072) * np.cos(10 * phi - 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_1_5_0_0():
    def func(k):
        return -47 / 211680 * k**10

    return func


def N_gg_1_5_0_0(theta, phi):
    return 1


def M_gg_1_5_0_1():
    def func(k):
        return -37 / 388080 * k**10

    return func


def N_gg_1_5_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_1_5_0_2():
    def func(k):
        return -349 / 31216185 * k**10

    return func


def N_gg_1_5_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_1_5_0_3():
    def func(k):
        return -4 / 27054027 * k**10

    return func


def N_gg_1_5_0_3(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * theta)
        + (819 / 256) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + 325 / 256
    )


def M_gg_1_5_2_0():
    def func(k):
        return -17 / 116424 * k**10

    return func


def N_gg_1_5_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_1_5_2_1():
    def func(k):
        return -17 / 116424 * k**10

    return func


def N_gg_1_5_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_1_5_2_2():
    def func(k):
        return -37 / 388080 * k**10

    return func


def N_gg_1_5_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_1_5_2_3():
    def func(k):
        return -229 / 6810804 * k**10

    return func


def N_gg_1_5_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_1_5_2_4():
    def func(k):
        return -229 / 6810804 * k**10

    return func


def N_gg_1_5_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_5_2_5():
    def func(k):
        return -349 / 31216185 * k**10

    return func


def N_gg_1_5_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_1_5_2_6():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_1_5_2_7():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_1_5_2_8():
    def func(k):
        return -4 / 27054027 * k**10

    return func


def N_gg_1_5_2_8(theta, phi):
    return (
        -6825 / 5632 * np.cos(2 * theta)
        + (819 / 2816) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + (819 / 512) * np.cos(2 * phi - 5 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 512) * np.cos(2 * phi + 5 * theta)
        - 2275 / 2816
    )


def M_gg_1_5_4_0():
    def func(k):
        return -7157 / 136216080 * k**10

    return func


def N_gg_1_5_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_1_5_4_1():
    def func(k):
        return -37 / 388080 * k**10

    return func


def N_gg_1_5_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_5_4_2():
    def func(k):
        return -229 / 6810804 * k**10

    return func


def N_gg_1_5_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_5_4_3():
    def func(k):
        return -53 / 9459450 * k**10

    return func


def N_gg_1_5_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_1_5_4_4():
    def func(k):
        return -7157 / 136216080 * k**10

    return func


def N_gg_1_5_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_1_5_4_5():
    def func(k):
        return -229 / 6810804 * k**10

    return func


def N_gg_1_5_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_5_4_6():
    def func(k):
        return -349 / 31216185 * k**10

    return func


def N_gg_1_5_4_6(theta, phi):
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


def M_gg_1_5_4_7():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_1_5_4_8():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_4_8(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (77175 / 73216) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (19845 / 6656) * np.cos(2 * phi + 3 * theta)
        + (6615 / 2048) * np.cos(2 * phi + 5 * theta)
        + (77175 / 1171456) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(4 * phi + 4 * theta)
        + (99225 / 8192) * np.cos(4 * phi + 6 * theta)
        + 694575 / 585728
    )


def M_gg_1_5_4_9():
    def func(k):
        return -53 / 9459450 * k**10

    return func


def N_gg_1_5_4_9(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_1_5_4_10():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_4_10(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_1_5_4_11():
    def func(k):
        return -4 / 27054027 * k**10

    return func


def N_gg_1_5_4_11(theta, phi):
    return (
        (1003275 / 382976) * np.cos(4 * phi)
        + (12285 / 69632) * np.cos(2 * theta)
        - 22113 / 11968 * np.cos(4 * theta)
        + (243243 / 69632) * np.cos(6 * theta)
        + (110565 / 34816) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(2 * phi + 5 * theta)
        + (85995 / 69632) * np.cos(4 * phi - 4 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (85995 / 69632) * np.cos(4 * phi + 4 * theta)
        + 61425 / 95744
    )


def M_gg_1_5_4_12():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_4_12(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (6615 / 2048) * np.cos(2 * phi - 5 * theta)
        + (19845 / 6656) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (77175 / 73216) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (99225 / 8192) * np.cos(4 * phi - 6 * theta)
        + (6615 / 2048) * np.cos(4 * phi - 4 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (77175 / 1171456) * np.cos(4 * phi + 2 * theta)
        + 694575 / 585728
    )


def M_gg_1_5_6_0():
    def func(k):
        return -173 / 18918900 * k**10

    return func


def N_gg_1_5_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_1_5_6_1():
    def func(k):
        return -229 / 6810804 * k**10

    return func


def N_gg_1_5_6_1(theta, phi):
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


def M_gg_1_5_6_2():
    def func(k):
        return -53 / 9459450 * k**10

    return func


def N_gg_1_5_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_1_5_6_3():
    def func(k):
        return -4 / 11486475 * k**10

    return func


def N_gg_1_5_6_3(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (9009 / 2048) * np.cos(4 * phi + 4 * theta)
        + (693 / 4096) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(6 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_1_5_6_4():
    def func(k):
        return -229 / 6810804 * k**10

    return func


def N_gg_1_5_6_4(theta, phi):
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


def M_gg_1_5_6_5():
    def func(k):
        return -349 / 31216185 * k**10

    return func


def N_gg_1_5_6_5(theta, phi):
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


def M_gg_1_5_6_6():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_6_6(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_1_5_6_7():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_6_7(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (1289925 / 856064) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        - 104895 / 77824 * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(2 * phi + 5 * theta)
        + (694575 / 856064) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        - 36855 / 19456 * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 6 * theta)
        + (19845 / 77824) * np.cos(6 * phi - theta)
        + (93555 / 77824) * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(6 * phi + 5 * theta)
        - 297675 / 428032
    )


def M_gg_1_5_6_8():
    def func(k):
        return -173 / 18918900 * k**10

    return func


def N_gg_1_5_6_8(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_1_5_6_9():
    def func(k):
        return -53 / 9459450 * k**10

    return func


def N_gg_1_5_6_9(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_1_5_6_10():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_6_10(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_1_5_6_11():
    def func(k):
        return -4 / 27054027 * k**10

    return func


def N_gg_1_5_6_11(theta, phi):
    return (
        -5589675 / 3638272 * np.cos(4 * phi)
        + (443625 / 661504) * np.cos(2 * theta)
        + (266175 / 909568) * np.cos(4 * theta)
        + (975975 / 661504) * np.cos(6 * theta)
        + (1863225 / 661504) * np.cos(2 * phi - 5 * theta)
        - 5589675 / 3638272 * np.cos(2 * phi - 3 * theta)
        + (443625 / 661504) * np.cos(2 * phi - theta)
        + (443625 / 661504) * np.cos(2 * phi + theta)
        - 5589675 / 3638272 * np.cos(2 * phi + 3 * theta)
        + (1863225 / 661504) * np.cos(2 * phi + 5 * theta)
        + (1863225 / 661504) * np.cos(4 * phi - 4 * theta)
        + (266175 / 909568) * np.cos(4 * phi - 2 * theta)
        + (266175 / 909568) * np.cos(4 * phi + 2 * theta)
        + (1863225 / 661504) * np.cos(4 * phi + 4 * theta)
        + (975975 / 661504) * np.cos(6 * phi - 3 * theta)
        + (1863225 / 661504) * np.cos(6 * phi - theta)
        + (1863225 / 661504) * np.cos(6 * phi + theta)
        + (975975 / 661504) * np.cos(6 * phi + 3 * theta)
        - 528125 / 909568
    )


def M_gg_1_5_6_12():
    def func(k):
        return -4 / 11486475 * k**10

    return func


def N_gg_1_5_6_12(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (9009 / 2048) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (45045 / 4096) * np.cos(6 * phi - 5 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (693 / 4096) * np.cos(6 * phi - theta)
        + 1225 / 1024
    )


def M_gg_1_5_6_13():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_6_13(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (257985 / 77824) * np.cos(2 * phi - 5 * theta)
        - 104895 / 77824 * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (1289925 / 856064) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 6 * theta)
        - 36855 / 19456 * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (694575 / 856064) * np.cos(4 * phi + 2 * theta)
        + (405405 / 77824) * np.cos(6 * phi - 5 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        + (93555 / 77824) * np.cos(6 * phi - theta)
        + (19845 / 77824) * np.cos(6 * phi + theta)
        - 297675 / 428032
    )


def M_gg_1_5_8_0():
    def func(k):
        return -1 / 1640925 * k**10

    return func


def N_gg_1_5_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi + theta)
        + (11781 / 4096) * np.cos(4 * phi + 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi + 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi + 4 * theta)
        + 20825 / 16384
    )


def M_gg_1_5_8_1():
    def func(k):
        return -53 / 9459450 * k**10

    return func


def N_gg_1_5_8_1(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_1_5_8_2():
    def func(k):
        return -4 / 11486475 * k**10

    return func


def N_gg_1_5_8_2(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (187425 / 77824) * np.cos(2 * phi - theta)
        - 26775 / 19456 * np.cos(2 * phi + theta)
        + (176715 / 77824) * np.cos(2 * phi + 3 * theta)
        - 11781 / 19456 * np.cos(4 * phi + 2 * theta)
        + (153153 / 77824) * np.cos(4 * phi + 4 * theta)
        + (153153 / 77824) * np.cos(6 * phi + theta)
        + (21879 / 19456) * np.cos(6 * phi + 3 * theta)
        + (109395 / 77824) * np.cos(6 * phi + 5 * theta)
        + (109395 / 77824) * np.cos(8 * phi + 2 * theta)
        + (546975 / 77824) * np.cos(8 * phi + 4 * theta)
        - 62475 / 77824
    )


def M_gg_1_5_8_3():
    def func(k):
        return -349 / 31216185 * k**10

    return func


def N_gg_1_5_8_3(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_1_5_8_4():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_1_5_8_5():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_8_5(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (5060475 / 2023424) * np.cos(2 * phi - 3 * theta)
        - 5060475 / 4046848 * np.cos(2 * phi - theta)
        + (2457945 / 4046848) * np.cos(2 * phi + theta)
        - 530145 / 4046848 * np.cos(2 * phi + 3 * theta)
        + (530145 / 311296) * np.cos(2 * phi + 5 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi - 2 * theta)
        - 954261 / 1011712 * np.cos(4 * phi + 2 * theta)
        + (530145 / 311296) * np.cos(4 * phi + 4 * theta)
        + (530145 / 622592) * np.cos(4 * phi + 6 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        - 590733 / 311296 * np.cos(6 * phi + 3 * theta)
        + (984555 / 311296) * np.cos(6 * phi + 5 * theta)
        + (984555 / 311296) * np.cos(8 * phi + 2 * theta)
        + (2953665 / 622592) * np.cos(8 * phi + 4 * theta)
        + 5060475 / 8093696
    )


def M_gg_1_5_8_6():
    def func(k):
        return -53 / 9459450 * k**10

    return func


def N_gg_1_5_8_6(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_1_5_8_7():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_8_7(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_1_5_8_8():
    def func(k):
        return -4 / 27054027 * k**10

    return func


def N_gg_1_5_8_8(theta, phi):
    return (
        (429975 / 311296) * np.cos(4 * phi)
        + (2395575 / 622592) * np.cos(8 * phi)
        - 16960125 / 13697024 * np.cos(2 * theta)
        + (12755925 / 6848512) * np.cos(4 * theta)
        + (525525 / 1245184) * np.cos(6 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (429975 / 311296) * np.cos(2 * phi + 5 * theta)
        + (1576575 / 622592) * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        + (1576575 / 622592) * np.cos(4 * phi + 4 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        - 266175 / 311296 * np.cos(6 * phi - theta)
        - 266175 / 311296 * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi - 2 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi + 2 * theta)
        + 1990625 / 3424256
    )


def M_gg_1_5_8_9():
    def func(k):
        return -1 / 1640925 * k**10

    return func


def N_gg_1_5_8_9(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi - theta)
        + (11781 / 4096) * np.cos(4 * phi - 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi - 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi - 4 * theta)
        + 20825 / 16384
    )


def M_gg_1_5_8_10():
    def func(k):
        return -4 / 11486475 * k**10

    return func


def N_gg_1_5_8_10(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (176715 / 77824) * np.cos(2 * phi - 3 * theta)
        - 26775 / 19456 * np.cos(2 * phi - theta)
        + (187425 / 77824) * np.cos(2 * phi + theta)
        + (153153 / 77824) * np.cos(4 * phi - 4 * theta)
        - 11781 / 19456 * np.cos(4 * phi - 2 * theta)
        + (109395 / 77824) * np.cos(6 * phi - 5 * theta)
        + (21879 / 19456) * np.cos(6 * phi - 3 * theta)
        + (153153 / 77824) * np.cos(6 * phi - theta)
        + (546975 / 77824) * np.cos(8 * phi - 4 * theta)
        + (109395 / 77824) * np.cos(8 * phi - 2 * theta)
        - 62475 / 77824
    )


def M_gg_1_5_8_11():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_8_11(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (530145 / 311296) * np.cos(2 * phi - 5 * theta)
        - 530145 / 4046848 * np.cos(2 * phi - 3 * theta)
        + (2457945 / 4046848) * np.cos(2 * phi - theta)
        - 5060475 / 4046848 * np.cos(2 * phi + theta)
        + (5060475 / 2023424) * np.cos(2 * phi + 3 * theta)
        + (530145 / 622592) * np.cos(4 * phi - 6 * theta)
        + (530145 / 311296) * np.cos(4 * phi - 4 * theta)
        - 954261 / 1011712 * np.cos(4 * phi - 2 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi + 2 * theta)
        + (984555 / 311296) * np.cos(6 * phi - 5 * theta)
        - 590733 / 311296 * np.cos(6 * phi - 3 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        + (2953665 / 622592) * np.cos(8 * phi - 4 * theta)
        + (984555 / 311296) * np.cos(8 * phi - 2 * theta)
        + 5060475 / 8093696
    )


def M_gg_1_5_10_0():
    def func(k):
        return -4 / 11486475 * k**10

    return func


def N_gg_1_5_10_0(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (1819125 / 1245184) * np.cos(2 * phi - theta)
        + (363825 / 155648) * np.cos(2 * phi + theta)
        + (363825 / 622592) * np.cos(2 * phi + 3 * theta)
        + (675675 / 311296) * np.cos(4 * phi + 2 * theta)
        + (96525 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 622592) * np.cos(6 * phi + theta)
        + (289575 / 155648) * np.cos(6 * phi + 3 * theta)
        + (289575 / 2490368) * np.cos(6 * phi + 5 * theta)
        + (1640925 / 311296) * np.cos(8 * phi + 2 * theta)
        + (1640925 / 1245184) * np.cos(8 * phi + 4 * theta)
        + (1640925 / 131072) * np.cos(10 * phi + 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_1_5_10_1():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_1_5_10_2():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_10_2(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (22920975 / 14319616) * np.cos(2 * phi - 3 * theta)
        + (363825 / 894976) * np.cos(2 * phi - theta)
        - 3274425 / 3579904 * np.cos(2 * phi + theta)
        + (12879405 / 7159808) * np.cos(2 * phi + 3 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi + 5 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi - 2 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi + 2 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (42567525 / 14319616) * np.cos(6 * phi - theta)
        - 13378365 / 7159808 * np.cos(6 * phi + theta)
        + (7123545 / 3579904) * np.cos(6 * phi + 3 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi + 5 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi + 2 * theta)
        + (44304975 / 14319616) * np.cos(8 * phi + 4 * theta)
        + (2297295 / 753664) * np.cos(10 * phi + theta)
        + (4922775 / 753664) * np.cos(10 * phi + 3 * theta)
        - 9823275 / 14319616
    )


def M_gg_1_5_10_3():
    def func(k):
        return -758 / 468242775 * k**10

    return func


def N_gg_1_5_10_3(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_1_5_10_4():
    def func(k):
        return -4 / 27054027 * k**10

    return func


def N_gg_1_5_10_4(theta, phi):
    return (
        -184459275 / 121716736 * np.cos(4 * phi)
        - 36891855 / 14319616 * np.cos(8 * phi)
        + (568856925 / 486866944) * np.cos(2 * theta)
        + (269800713 / 243433472) * np.cos(4 * theta)
        + (35756721 / 486866944) * np.cos(6 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi - 5 * theta)
        + (854188335 / 486866944) * np.cos(2 * phi - 3 * theta)
        - 99324225 / 243433472 * np.cos(2 * phi - theta)
        - 99324225 / 243433472 * np.cos(2 * phi + theta)
        + (854188335 / 486866944) * np.cos(2 * phi + 3 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi + 5 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi + 2 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi + 4 * theta)
        + (553377825 / 243433472) * np.cos(6 * phi - 3 * theta)
        - 110675565 / 486866944 * np.cos(6 * phi - theta)
        - 110675565 / 486866944 * np.cos(6 * phi + theta)
        + (553377825 / 243433472) * np.cos(6 * phi + 3 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi - 2 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi + 2 * theta)
        + (7378371 / 1507328) * np.cos(10 * phi - theta)
        + (7378371 / 1507328) * np.cos(10 * phi + theta)
        - 81265275 / 121716736
    )


def M_gg_1_5_10_5():
    def func(k):
        return -4 / 11486475 * k**10

    return func


def N_gg_1_5_10_5(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (363825 / 622592) * np.cos(2 * phi - 3 * theta)
        + (363825 / 155648) * np.cos(2 * phi - theta)
        + (1819125 / 1245184) * np.cos(2 * phi + theta)
        + (96525 / 311296) * np.cos(4 * phi - 4 * theta)
        + (675675 / 311296) * np.cos(4 * phi - 2 * theta)
        + (289575 / 2490368) * np.cos(6 * phi - 5 * theta)
        + (289575 / 155648) * np.cos(6 * phi - 3 * theta)
        + (2027025 / 622592) * np.cos(6 * phi - theta)
        + (1640925 / 1245184) * np.cos(8 * phi - 4 * theta)
        + (1640925 / 311296) * np.cos(8 * phi - 2 * theta)
        + (1640925 / 131072) * np.cos(10 * phi - 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_1_5_10_6():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_10_6(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi - 5 * theta)
        + (12879405 / 7159808) * np.cos(2 * phi - 3 * theta)
        - 3274425 / 3579904 * np.cos(2 * phi - theta)
        + (363825 / 894976) * np.cos(2 * phi + theta)
        + (22920975 / 14319616) * np.cos(2 * phi + 3 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi - 6 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi - 4 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi - 2 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi + 2 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi - 5 * theta)
        + (7123545 / 3579904) * np.cos(6 * phi - 3 * theta)
        - 13378365 / 7159808 * np.cos(6 * phi - theta)
        + (42567525 / 14319616) * np.cos(6 * phi + theta)
        + (44304975 / 14319616) * np.cos(8 * phi - 4 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi - 2 * theta)
        + (4922775 / 753664) * np.cos(10 * phi - 3 * theta)
        + (2297295 / 753664) * np.cos(10 * phi - theta)
        - 9823275 / 14319616
    )


def M_gg_1_5_12_0():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_12_0(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (42567525 / 114556928) * np.cos(2 * phi - 3 * theta)
        + (212837625 / 114556928) * np.cos(2 * phi - theta)
        + (127702575 / 57278464) * np.cos(2 * phi + theta)
        + (42567525 / 57278464) * np.cos(2 * phi + 3 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi + 5 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi - 2 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi + 2 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi + 4 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi + 6 * theta)
        + (172297125 / 114556928) * np.cos(6 * phi - theta)
        + (172297125 / 57278464) * np.cos(6 * phi + theta)
        + (73841625 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi + 5 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi + 2 * theta)
        + (14768325 / 24117248) * np.cos(8 * phi + 4 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi + theta)
        + (34459425 / 12058624) * np.cos(10 * phi + 3 * theta)
        + (34459425 / 2097152) * np.cos(12 * phi + 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_1_5_12_1():
    def func(k):
        return -4 / 27054027 * k**10

    return func


def N_gg_1_5_12_1(theta, phi):
    return (
        (10145260125 / 3894935552) * np.cos(4 * phi)
        + (11594583 / 3014656) * np.cos(8 * phi)
        + (9018009 / 524288) * np.cos(12 * phi)
        + (5150670525 / 3894935552) * np.cos(2 * theta)
        + (206026821 / 973733888) * np.cos(4 * theta)
        + (22891869 / 3894935552) * np.cos(6 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi - 5 * theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi - 3 * theta)
        + (2029052025 / 973733888) * np.cos(2 * phi - theta)
        + (2029052025 / 973733888) * np.cos(2 * phi + theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi + 3 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi + 5 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi - 4 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi - 2 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi + 2 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi + 4 * theta)
        + (32207175 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (289864575 / 114556928) * np.cos(6 * phi - theta)
        + (289864575 / 114556928) * np.cos(6 * phi + theta)
        + (32207175 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (27054027 / 6029312) * np.cos(10 * phi - theta)
        + (27054027 / 6029312) * np.cos(10 * phi + theta)
        + 572296725 / 486866944
    )


def M_gg_1_5_12_2():
    def func(k):
        return -8 / 103378275 * k**10

    return func


def N_gg_1_5_12_2(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi - 5 * theta)
        + (42567525 / 57278464) * np.cos(2 * phi - 3 * theta)
        + (127702575 / 57278464) * np.cos(2 * phi - theta)
        + (212837625 / 114556928) * np.cos(2 * phi + theta)
        + (42567525 / 114556928) * np.cos(2 * phi + 3 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi - 6 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi - 4 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi + 2 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi - 5 * theta)
        + (73841625 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (172297125 / 57278464) * np.cos(6 * phi - theta)
        + (172297125 / 114556928) * np.cos(6 * phi + theta)
        + (14768325 / 24117248) * np.cos(8 * phi - 4 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi - 2 * theta)
        + (34459425 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi - theta)
        + (34459425 / 2097152) * np.cos(12 * phi - 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_1_6_0_0():
    def func(k):
        return (1 / 72576) * k**12

    return func


def N_gg_1_6_0_0(theta, phi):
    return 1


def M_gg_1_6_0_1():
    def func(k):
        return (1 / 149688) * k**12

    return func


def N_gg_1_6_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_1_6_0_2():
    def func(k):
        return (1 / 891891) * k**12

    return func


def N_gg_1_6_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_1_6_0_3():
    def func(k):
        return (8 / 173918745) * k**12

    return func


def N_gg_1_6_0_3(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * theta)
        + (819 / 256) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + 325 / 256
    )


def M_gg_1_6_2_0():
    def func(k):
        return (23 / 2395008) * k**12

    return func


def N_gg_1_6_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_1_6_2_1():
    def func(k):
        return (23 / 2395008) * k**12

    return func


def N_gg_1_6_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_1_6_2_2():
    def func(k):
        return (1 / 149688) * k**12

    return func


def N_gg_1_6_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_1_6_2_3():
    def func(k):
        return (59 / 21405384) * k**12

    return func


def N_gg_1_6_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_1_6_2_4():
    def func(k):
        return (59 / 21405384) * k**12

    return func


def N_gg_1_6_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_6_2_5():
    def func(k):
        return (1 / 891891) * k**12

    return func


def N_gg_1_6_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_1_6_2_6():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_1_6_2_7():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_1_6_2_8():
    def func(k):
        return (8 / 173918745) * k**12

    return func


def N_gg_1_6_2_8(theta, phi):
    return (
        -6825 / 5632 * np.cos(2 * theta)
        + (819 / 2816) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + (819 / 512) * np.cos(2 * phi - 5 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 512) * np.cos(2 * phi + 5 * theta)
        - 2275 / 2816
    )


def M_gg_1_6_2_9():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_2_9(theta, phi):
    return (
        (4725 / 2048) * np.cos(2 * theta)
        + (2079 / 1024) * np.cos(4 * theta)
        + (3003 / 2048) * np.cos(6 * theta)
        + (693 / 4096) * np.cos(2 * phi - 5 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (9009 / 2048) * np.cos(2 * phi + 5 * theta)
        + (45045 / 4096) * np.cos(2 * phi + 7 * theta)
        + 1225 / 1024
    )


def M_gg_1_6_2_10():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_2_10(theta, phi):
    return (
        (4725 / 2048) * np.cos(2 * theta)
        + (2079 / 1024) * np.cos(4 * theta)
        + (3003 / 2048) * np.cos(6 * theta)
        + (45045 / 4096) * np.cos(2 * phi - 7 * theta)
        + (9009 / 2048) * np.cos(2 * phi - 5 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (693 / 4096) * np.cos(2 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_1_6_4_0():
    def func(k):
        return (31 / 7783776) * k**12

    return func


def N_gg_1_6_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_1_6_4_1():
    def func(k):
        return (1 / 149688) * k**12

    return func


def N_gg_1_6_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_1_6_4_2():
    def func(k):
        return (59 / 21405384) * k**12

    return func


def N_gg_1_6_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_6_4_3():
    def func(k):
        return (8 / 13378365) * k**12

    return func


def N_gg_1_6_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_1_6_4_4():
    def func(k):
        return (31 / 7783776) * k**12

    return func


def N_gg_1_6_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_1_6_4_5():
    def func(k):
        return (59 / 21405384) * k**12

    return func


def N_gg_1_6_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_1_6_4_6():
    def func(k):
        return (1 / 891891) * k**12

    return func


def N_gg_1_6_4_6(theta, phi):
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


def M_gg_1_6_4_7():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_1_6_4_8():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_4_8(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (77175 / 73216) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (19845 / 6656) * np.cos(2 * phi + 3 * theta)
        + (6615 / 2048) * np.cos(2 * phi + 5 * theta)
        + (77175 / 1171456) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(4 * phi + 4 * theta)
        + (99225 / 8192) * np.cos(4 * phi + 6 * theta)
        + 694575 / 585728
    )


def M_gg_1_6_4_9():
    def func(k):
        return (8 / 13378365) * k**12

    return func


def N_gg_1_6_4_9(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_1_6_4_10():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_4_10(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_1_6_4_11():
    def func(k):
        return (8 / 173918745) * k**12

    return func


def N_gg_1_6_4_11(theta, phi):
    return (
        (1003275 / 382976) * np.cos(4 * phi)
        + (12285 / 69632) * np.cos(2 * theta)
        - 22113 / 11968 * np.cos(4 * theta)
        + (243243 / 69632) * np.cos(6 * theta)
        + (110565 / 34816) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(2 * phi + 5 * theta)
        + (85995 / 69632) * np.cos(4 * phi - 4 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (85995 / 69632) * np.cos(4 * phi + 4 * theta)
        + 61425 / 95744
    )


def M_gg_1_6_4_12():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_4_12(theta, phi):
    return (
        (694575 / 428032) * np.cos(4 * phi)
        - 552825 / 856064 * np.cos(2 * theta)
        + (49329 / 38912) * np.cos(4 * theta)
        + (243243 / 77824) * np.cos(6 * theta)
        + (93555 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        + (1289925 / 856064) * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        - 104895 / 77824 * np.cos(2 * phi + 3 * theta)
        - 36855 / 19456 * np.cos(2 * phi + 5 * theta)
        + (405405 / 77824) * np.cos(2 * phi + 7 * theta)
        + (19845 / 77824) * np.cos(4 * phi - 4 * theta)
        + (694575 / 856064) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 6 * theta)
        - 297675 / 428032
    )


def M_gg_1_6_4_13():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_4_13(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (6615 / 2048) * np.cos(2 * phi - 5 * theta)
        + (19845 / 6656) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (77175 / 73216) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (99225 / 8192) * np.cos(4 * phi - 6 * theta)
        + (6615 / 2048) * np.cos(4 * phi - 4 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (77175 / 1171456) * np.cos(4 * phi + 2 * theta)
        + 694575 / 585728
    )


def M_gg_1_6_4_14():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_4_14(theta, phi):
    return (
        (694575 / 428032) * np.cos(4 * phi)
        - 552825 / 856064 * np.cos(2 * theta)
        + (49329 / 38912) * np.cos(4 * theta)
        + (243243 / 77824) * np.cos(6 * theta)
        + (405405 / 77824) * np.cos(2 * phi - 7 * theta)
        - 36855 / 19456 * np.cos(2 * phi - 5 * theta)
        - 104895 / 77824 * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        + (1289925 / 856064) * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(2 * phi + 5 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 6 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 4 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (694575 / 856064) * np.cos(4 * phi + 2 * theta)
        + (19845 / 77824) * np.cos(4 * phi + 4 * theta)
        - 297675 / 428032
    )


def M_gg_1_6_6_0():
    def func(k):
        return (17 / 19459440) * k**12

    return func


def N_gg_1_6_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_1_6_6_1():
    def func(k):
        return (59 / 21405384) * k**12

    return func


def N_gg_1_6_6_1(theta, phi):
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


def M_gg_1_6_6_2():
    def func(k):
        return (8 / 13378365) * k**12

    return func


def N_gg_1_6_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_1_6_6_3():
    def func(k):
        return (1 / 20675655) * k**12

    return func


def N_gg_1_6_6_3(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (9009 / 2048) * np.cos(4 * phi + 4 * theta)
        + (693 / 4096) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(6 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_1_6_6_4():
    def func(k):
        return (59 / 21405384) * k**12

    return func


def N_gg_1_6_6_4(theta, phi):
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


def M_gg_1_6_6_5():
    def func(k):
        return (1 / 891891) * k**12

    return func


def N_gg_1_6_6_5(theta, phi):
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


def M_gg_1_6_6_6():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_6_6(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_1_6_6_7():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_6_7(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (1289925 / 856064) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        - 104895 / 77824 * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(2 * phi + 5 * theta)
        + (694575 / 856064) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        - 36855 / 19456 * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 6 * theta)
        + (19845 / 77824) * np.cos(6 * phi - theta)
        + (93555 / 77824) * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(6 * phi + 5 * theta)
        - 297675 / 428032
    )


def M_gg_1_6_6_8():
    def func(k):
        return (17 / 19459440) * k**12

    return func


def N_gg_1_6_6_8(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_1_6_6_9():
    def func(k):
        return (8 / 13378365) * k**12

    return func


def N_gg_1_6_6_9(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_1_6_6_10():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_6_10(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_1_6_6_11():
    def func(k):
        return (8 / 173918745) * k**12

    return func


def N_gg_1_6_6_11(theta, phi):
    return (
        -5589675 / 3638272 * np.cos(4 * phi)
        + (443625 / 661504) * np.cos(2 * theta)
        + (266175 / 909568) * np.cos(4 * theta)
        + (975975 / 661504) * np.cos(6 * theta)
        + (1863225 / 661504) * np.cos(2 * phi - 5 * theta)
        - 5589675 / 3638272 * np.cos(2 * phi - 3 * theta)
        + (443625 / 661504) * np.cos(2 * phi - theta)
        + (443625 / 661504) * np.cos(2 * phi + theta)
        - 5589675 / 3638272 * np.cos(2 * phi + 3 * theta)
        + (1863225 / 661504) * np.cos(2 * phi + 5 * theta)
        + (1863225 / 661504) * np.cos(4 * phi - 4 * theta)
        + (266175 / 909568) * np.cos(4 * phi - 2 * theta)
        + (266175 / 909568) * np.cos(4 * phi + 2 * theta)
        + (1863225 / 661504) * np.cos(4 * phi + 4 * theta)
        + (975975 / 661504) * np.cos(6 * phi - 3 * theta)
        + (1863225 / 661504) * np.cos(6 * phi - theta)
        + (1863225 / 661504) * np.cos(6 * phi + theta)
        + (975975 / 661504) * np.cos(6 * phi + 3 * theta)
        - 528125 / 909568
    )


def M_gg_1_6_6_12():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_6_12(theta, phi):
    return (
        (429975 / 856064) * np.cos(4 * phi)
        - 716625 / 3424256 * np.cos(2 * theta)
        - 429975 / 311296 * np.cos(4 * theta)
        + (975975 / 311296) * np.cos(6 * theta)
        + (1576575 / 622592) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 16960125 / 13697024 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 311296) * np.cos(2 * phi + 3 * theta)
        - 266175 / 311296 * np.cos(2 * phi + 5 * theta)
        + (2927925 / 1245184) * np.cos(2 * phi + 7 * theta)
        + (429975 / 311296) * np.cos(4 * phi - 4 * theta)
        + (12755925 / 6848512) * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        - 266175 / 311296 * np.cos(4 * phi + 4 * theta)
        + (2395575 / 622592) * np.cos(4 * phi + 6 * theta)
        + (525525 / 1245184) * np.cos(6 * phi - 3 * theta)
        + (429975 / 311296) * np.cos(6 * phi - theta)
        + (1576575 / 622592) * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(6 * phi + 5 * theta)
        + 1990625 / 3424256
    )


def M_gg_1_6_6_13():
    def func(k):
        return (1 / 20675655) * k**12

    return func


def N_gg_1_6_6_13(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (9009 / 2048) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (45045 / 4096) * np.cos(6 * phi - 5 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (693 / 4096) * np.cos(6 * phi - theta)
        + 1225 / 1024
    )


def M_gg_1_6_6_14():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_6_14(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (257985 / 77824) * np.cos(2 * phi - 5 * theta)
        - 104895 / 77824 * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (1289925 / 856064) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 6 * theta)
        - 36855 / 19456 * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (694575 / 856064) * np.cos(4 * phi + 2 * theta)
        + (405405 / 77824) * np.cos(6 * phi - 5 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        + (93555 / 77824) * np.cos(6 * phi - theta)
        + (19845 / 77824) * np.cos(6 * phi + theta)
        - 297675 / 428032
    )


def M_gg_1_6_6_15():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_6_15(theta, phi):
    return (
        (429975 / 856064) * np.cos(4 * phi)
        - 716625 / 3424256 * np.cos(2 * theta)
        - 429975 / 311296 * np.cos(4 * theta)
        + (975975 / 311296) * np.cos(6 * theta)
        + (2927925 / 1245184) * np.cos(2 * phi - 7 * theta)
        - 266175 / 311296 * np.cos(2 * phi - 5 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 16960125 / 13697024 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (1576575 / 622592) * np.cos(2 * phi + 5 * theta)
        + (2395575 / 622592) * np.cos(4 * phi - 6 * theta)
        - 266175 / 311296 * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        + (12755925 / 6848512) * np.cos(4 * phi + 2 * theta)
        + (429975 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2927925 / 1245184) * np.cos(6 * phi - 5 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        + (1576575 / 622592) * np.cos(6 * phi - theta)
        + (429975 / 311296) * np.cos(6 * phi + theta)
        + (525525 / 1245184) * np.cos(6 * phi + 3 * theta)
        + 1990625 / 3424256
    )


def M_gg_1_6_8_0():
    def func(k):
        return (1 / 13783770) * k**12

    return func


def N_gg_1_6_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi + theta)
        + (11781 / 4096) * np.cos(4 * phi + 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi + 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi + 4 * theta)
        + 20825 / 16384
    )


def M_gg_1_6_8_1():
    def func(k):
        return (8 / 13378365) * k**12

    return func


def N_gg_1_6_8_1(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_1_6_8_2():
    def func(k):
        return (1 / 20675655) * k**12

    return func


def N_gg_1_6_8_2(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (187425 / 77824) * np.cos(2 * phi - theta)
        - 26775 / 19456 * np.cos(2 * phi + theta)
        + (176715 / 77824) * np.cos(2 * phi + 3 * theta)
        - 11781 / 19456 * np.cos(4 * phi + 2 * theta)
        + (153153 / 77824) * np.cos(4 * phi + 4 * theta)
        + (153153 / 77824) * np.cos(6 * phi + theta)
        + (21879 / 19456) * np.cos(6 * phi + 3 * theta)
        + (109395 / 77824) * np.cos(6 * phi + 5 * theta)
        + (109395 / 77824) * np.cos(8 * phi + 2 * theta)
        + (546975 / 77824) * np.cos(8 * phi + 4 * theta)
        - 62475 / 77824
    )


def M_gg_1_6_8_3():
    def func(k):
        return (1 / 891891) * k**12

    return func


def N_gg_1_6_8_3(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_1_6_8_4():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_1_6_8_5():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_8_5(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (5060475 / 2023424) * np.cos(2 * phi - 3 * theta)
        - 5060475 / 4046848 * np.cos(2 * phi - theta)
        + (2457945 / 4046848) * np.cos(2 * phi + theta)
        - 530145 / 4046848 * np.cos(2 * phi + 3 * theta)
        + (530145 / 311296) * np.cos(2 * phi + 5 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi - 2 * theta)
        - 954261 / 1011712 * np.cos(4 * phi + 2 * theta)
        + (530145 / 311296) * np.cos(4 * phi + 4 * theta)
        + (530145 / 622592) * np.cos(4 * phi + 6 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        - 590733 / 311296 * np.cos(6 * phi + 3 * theta)
        + (984555 / 311296) * np.cos(6 * phi + 5 * theta)
        + (984555 / 311296) * np.cos(8 * phi + 2 * theta)
        + (2953665 / 622592) * np.cos(8 * phi + 4 * theta)
        + 5060475 / 8093696
    )


def M_gg_1_6_8_6():
    def func(k):
        return (8 / 13378365) * k**12

    return func


def N_gg_1_6_8_6(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_1_6_8_7():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_8_7(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_1_6_8_8():
    def func(k):
        return (8 / 173918745) * k**12

    return func


def N_gg_1_6_8_8(theta, phi):
    return (
        (429975 / 311296) * np.cos(4 * phi)
        + (2395575 / 622592) * np.cos(8 * phi)
        - 16960125 / 13697024 * np.cos(2 * theta)
        + (12755925 / 6848512) * np.cos(4 * theta)
        + (525525 / 1245184) * np.cos(6 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (429975 / 311296) * np.cos(2 * phi + 5 * theta)
        + (1576575 / 622592) * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        + (1576575 / 622592) * np.cos(4 * phi + 4 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        - 266175 / 311296 * np.cos(6 * phi - theta)
        - 266175 / 311296 * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi - 2 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi + 2 * theta)
        + 1990625 / 3424256
    )


def M_gg_1_6_8_9():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_8_9(theta, phi):
    return (
        -6185025 / 7159808 * np.cos(4 * phi)
        + (34459425 / 14319616) * np.cos(8 * phi)
        + (12182625 / 14319616) * np.cos(2 * theta)
        - 6185025 / 14319616 * np.cos(4 * theta)
        + (26801775 / 14319616) * np.cos(6 * theta)
        + (18555075 / 7159808) * np.cos(2 * phi - 5 * theta)
        - 5060475 / 3579904 * np.cos(2 * phi - 3 * theta)
        + (12182625 / 14319616) * np.cos(2 * phi - theta)
        + (133875 / 7159808) * np.cos(2 * phi + theta)
        - 6185025 / 7159808 * np.cos(2 * phi + 3 * theta)
        + (11486475 / 7159808) * np.cos(2 * phi + 5 * theta)
        + (11486475 / 14319616) * np.cos(2 * phi + 7 * theta)
        + (18555075 / 7159808) * np.cos(4 * phi - 4 * theta)
        - 6185025 / 14319616 * np.cos(4 * phi - 2 * theta)
        + (294525 / 223744) * np.cos(4 * phi + 2 * theta)
        - 3828825 / 3579904 * np.cos(4 * phi + 4 * theta)
        + (34459425 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (26801775 / 14319616) * np.cos(6 * phi - 3 * theta)
        + (11486475 / 7159808) * np.cos(6 * phi - theta)
        - 3828825 / 3579904 * np.cos(6 * phi + theta)
        - 7110675 / 7159808 * np.cos(6 * phi + 3 * theta)
        + (49774725 / 14319616) * np.cos(6 * phi + 5 * theta)
        + (11486475 / 14319616) * np.cos(8 * phi - 2 * theta)
        + (49774725 / 14319616) * np.cos(8 * phi + 2 * theta)
        + (35553375 / 14319616) * np.cos(8 * phi + 4 * theta)
        - 7809375 / 14319616
    )


def M_gg_1_6_8_10():
    def func(k):
        return (1 / 13783770) * k**12

    return func


def N_gg_1_6_8_10(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi - theta)
        + (11781 / 4096) * np.cos(4 * phi - 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi - 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi - 4 * theta)
        + 20825 / 16384
    )


def M_gg_1_6_8_11():
    def func(k):
        return (1 / 20675655) * k**12

    return func


def N_gg_1_6_8_11(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (176715 / 77824) * np.cos(2 * phi - 3 * theta)
        - 26775 / 19456 * np.cos(2 * phi - theta)
        + (187425 / 77824) * np.cos(2 * phi + theta)
        + (153153 / 77824) * np.cos(4 * phi - 4 * theta)
        - 11781 / 19456 * np.cos(4 * phi - 2 * theta)
        + (109395 / 77824) * np.cos(6 * phi - 5 * theta)
        + (21879 / 19456) * np.cos(6 * phi - 3 * theta)
        + (153153 / 77824) * np.cos(6 * phi - theta)
        + (546975 / 77824) * np.cos(8 * phi - 4 * theta)
        + (109395 / 77824) * np.cos(8 * phi - 2 * theta)
        - 62475 / 77824
    )


def M_gg_1_6_8_12():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_8_12(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (530145 / 311296) * np.cos(2 * phi - 5 * theta)
        - 530145 / 4046848 * np.cos(2 * phi - 3 * theta)
        + (2457945 / 4046848) * np.cos(2 * phi - theta)
        - 5060475 / 4046848 * np.cos(2 * phi + theta)
        + (5060475 / 2023424) * np.cos(2 * phi + 3 * theta)
        + (530145 / 622592) * np.cos(4 * phi - 6 * theta)
        + (530145 / 311296) * np.cos(4 * phi - 4 * theta)
        - 954261 / 1011712 * np.cos(4 * phi - 2 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi + 2 * theta)
        + (984555 / 311296) * np.cos(6 * phi - 5 * theta)
        - 590733 / 311296 * np.cos(6 * phi - 3 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        + (2953665 / 622592) * np.cos(8 * phi - 4 * theta)
        + (984555 / 311296) * np.cos(8 * phi - 2 * theta)
        + 5060475 / 8093696
    )


def M_gg_1_6_8_13():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_8_13(theta, phi):
    return (
        -6185025 / 7159808 * np.cos(4 * phi)
        + (34459425 / 14319616) * np.cos(8 * phi)
        + (12182625 / 14319616) * np.cos(2 * theta)
        - 6185025 / 14319616 * np.cos(4 * theta)
        + (26801775 / 14319616) * np.cos(6 * theta)
        + (11486475 / 14319616) * np.cos(2 * phi - 7 * theta)
        + (11486475 / 7159808) * np.cos(2 * phi - 5 * theta)
        - 6185025 / 7159808 * np.cos(2 * phi - 3 * theta)
        + (133875 / 7159808) * np.cos(2 * phi - theta)
        + (12182625 / 14319616) * np.cos(2 * phi + theta)
        - 5060475 / 3579904 * np.cos(2 * phi + 3 * theta)
        + (18555075 / 7159808) * np.cos(2 * phi + 5 * theta)
        + (34459425 / 14319616) * np.cos(4 * phi - 6 * theta)
        - 3828825 / 3579904 * np.cos(4 * phi - 4 * theta)
        + (294525 / 223744) * np.cos(4 * phi - 2 * theta)
        - 6185025 / 14319616 * np.cos(4 * phi + 2 * theta)
        + (18555075 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (49774725 / 14319616) * np.cos(6 * phi - 5 * theta)
        - 7110675 / 7159808 * np.cos(6 * phi - 3 * theta)
        - 3828825 / 3579904 * np.cos(6 * phi - theta)
        + (11486475 / 7159808) * np.cos(6 * phi + theta)
        + (26801775 / 14319616) * np.cos(6 * phi + 3 * theta)
        + (35553375 / 14319616) * np.cos(8 * phi - 4 * theta)
        + (49774725 / 14319616) * np.cos(8 * phi - 2 * theta)
        + (11486475 / 14319616) * np.cos(8 * phi + 2 * theta)
        - 7809375 / 14319616
    )


def M_gg_1_6_10_0():
    def func(k):
        return (1 / 20675655) * k**12

    return func


def N_gg_1_6_10_0(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (1819125 / 1245184) * np.cos(2 * phi - theta)
        + (363825 / 155648) * np.cos(2 * phi + theta)
        + (363825 / 622592) * np.cos(2 * phi + 3 * theta)
        + (675675 / 311296) * np.cos(4 * phi + 2 * theta)
        + (96525 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 622592) * np.cos(6 * phi + theta)
        + (289575 / 155648) * np.cos(6 * phi + 3 * theta)
        + (289575 / 2490368) * np.cos(6 * phi + 5 * theta)
        + (1640925 / 311296) * np.cos(8 * phi + 2 * theta)
        + (1640925 / 1245184) * np.cos(8 * phi + 4 * theta)
        + (1640925 / 131072) * np.cos(10 * phi + 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_1_6_10_1():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_1_6_10_2():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_10_2(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (22920975 / 14319616) * np.cos(2 * phi - 3 * theta)
        + (363825 / 894976) * np.cos(2 * phi - theta)
        - 3274425 / 3579904 * np.cos(2 * phi + theta)
        + (12879405 / 7159808) * np.cos(2 * phi + 3 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi + 5 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi - 2 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi + 2 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (42567525 / 14319616) * np.cos(6 * phi - theta)
        - 13378365 / 7159808 * np.cos(6 * phi + theta)
        + (7123545 / 3579904) * np.cos(6 * phi + 3 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi + 5 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi + 2 * theta)
        + (44304975 / 14319616) * np.cos(8 * phi + 4 * theta)
        + (2297295 / 753664) * np.cos(10 * phi + theta)
        + (4922775 / 753664) * np.cos(10 * phi + 3 * theta)
        - 9823275 / 14319616
    )


def M_gg_1_6_10_3():
    def func(k):
        return (41 / 173918745) * k**12

    return func


def N_gg_1_6_10_3(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_1_6_10_4():
    def func(k):
        return (8 / 173918745) * k**12

    return func


def N_gg_1_6_10_4(theta, phi):
    return (
        -184459275 / 121716736 * np.cos(4 * phi)
        - 36891855 / 14319616 * np.cos(8 * phi)
        + (568856925 / 486866944) * np.cos(2 * theta)
        + (269800713 / 243433472) * np.cos(4 * theta)
        + (35756721 / 486866944) * np.cos(6 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi - 5 * theta)
        + (854188335 / 486866944) * np.cos(2 * phi - 3 * theta)
        - 99324225 / 243433472 * np.cos(2 * phi - theta)
        - 99324225 / 243433472 * np.cos(2 * phi + theta)
        + (854188335 / 486866944) * np.cos(2 * phi + 3 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi + 5 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi + 2 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi + 4 * theta)
        + (553377825 / 243433472) * np.cos(6 * phi - 3 * theta)
        - 110675565 / 486866944 * np.cos(6 * phi - theta)
        - 110675565 / 486866944 * np.cos(6 * phi + theta)
        + (553377825 / 243433472) * np.cos(6 * phi + 3 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi - 2 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi + 2 * theta)
        + (7378371 / 1507328) * np.cos(10 * phi - theta)
        + (7378371 / 1507328) * np.cos(10 * phi + theta)
        - 81265275 / 121716736
    )


def M_gg_1_6_10_5():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_10_5(theta, phi):
    return (
        (945945 / 894976) * np.cos(4 * phi)
        + (9648639 / 28639232) * np.cos(8 * phi)
        - 138442689 / 114556928 * np.cos(2 * theta)
        + (112771197 / 71598080) * np.cos(4 * theta)
        + (393323931 / 572784640) * np.cos(6 * theta)
        + (332812557 / 229113856) * np.cos(2 * phi - 5 * theta)
        + (48592467 / 114556928) * np.cos(2 * phi - 3 * theta)
        - 89137125 / 229113856 * np.cos(2 * phi - theta)
        + (14771295 / 57278464) * np.cos(2 * phi + theta)
        - 79968735 / 229113856 * np.cos(2 * phi + 3 * theta)
        + (194675481 / 114556928) * np.cos(2 * phi + 5 * theta)
        + (43702659 / 229113856) * np.cos(2 * phi + 7 * theta)
        + (131107977 / 57278464) * np.cos(4 * phi - 4 * theta)
        - 59594535 / 57278464 * np.cos(4 * phi - 2 * theta)
        - 34999965 / 28639232 * np.cos(4 * phi + 2 * theta)
        + (86080995 / 57278464) * np.cos(4 * phi + 4 * theta)
        + (51648597 / 57278464) * np.cos(4 * phi + 6 * theta)
        + (655539885 / 229113856) * np.cos(6 * phi - 3 * theta)
        - 171972801 / 114556928 * np.cos(6 * phi - theta)
        + (218513295 / 229113856) * np.cos(6 * phi + theta)
        - 36891855 / 57278464 * np.cos(6 * phi + 3 * theta)
        + (258242985 / 114556928) * np.cos(6 * phi + 5 * theta)
        + (318405087 / 114556928) * np.cos(8 * phi - 2 * theta)
        - 209053845 / 114556928 * np.cos(8 * phi + 2 * theta)
        + (209053845 / 57278464) * np.cos(8 * phi + 4 * theta)
        + (106135029 / 60293120) * np.cos(10 * phi - theta)
        + (125432307 / 30146560) * np.cos(10 * phi + theta)
        + (41810769 / 12058624) * np.cos(10 * phi + 3 * theta)
        + 32089365 / 57278464
    )


def M_gg_1_6_10_6():
    def func(k):
        return (1 / 20675655) * k**12

    return func


def N_gg_1_6_10_6(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (363825 / 622592) * np.cos(2 * phi - 3 * theta)
        + (363825 / 155648) * np.cos(2 * phi - theta)
        + (1819125 / 1245184) * np.cos(2 * phi + theta)
        + (96525 / 311296) * np.cos(4 * phi - 4 * theta)
        + (675675 / 311296) * np.cos(4 * phi - 2 * theta)
        + (289575 / 2490368) * np.cos(6 * phi - 5 * theta)
        + (289575 / 155648) * np.cos(6 * phi - 3 * theta)
        + (2027025 / 622592) * np.cos(6 * phi - theta)
        + (1640925 / 1245184) * np.cos(8 * phi - 4 * theta)
        + (1640925 / 311296) * np.cos(8 * phi - 2 * theta)
        + (1640925 / 131072) * np.cos(10 * phi - 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_1_6_10_7():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_10_7(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi - 5 * theta)
        + (12879405 / 7159808) * np.cos(2 * phi - 3 * theta)
        - 3274425 / 3579904 * np.cos(2 * phi - theta)
        + (363825 / 894976) * np.cos(2 * phi + theta)
        + (22920975 / 14319616) * np.cos(2 * phi + 3 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi - 6 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi - 4 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi - 2 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi + 2 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi - 5 * theta)
        + (7123545 / 3579904) * np.cos(6 * phi - 3 * theta)
        - 13378365 / 7159808 * np.cos(6 * phi - theta)
        + (42567525 / 14319616) * np.cos(6 * phi + theta)
        + (44304975 / 14319616) * np.cos(8 * phi - 4 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi - 2 * theta)
        + (4922775 / 753664) * np.cos(10 * phi - 3 * theta)
        + (2297295 / 753664) * np.cos(10 * phi - theta)
        - 9823275 / 14319616
    )


def M_gg_1_6_10_8():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_10_8(theta, phi):
    return (
        (945945 / 894976) * np.cos(4 * phi)
        + (9648639 / 28639232) * np.cos(8 * phi)
        - 138442689 / 114556928 * np.cos(2 * theta)
        + (112771197 / 71598080) * np.cos(4 * theta)
        + (393323931 / 572784640) * np.cos(6 * theta)
        + (43702659 / 229113856) * np.cos(2 * phi - 7 * theta)
        + (194675481 / 114556928) * np.cos(2 * phi - 5 * theta)
        - 79968735 / 229113856 * np.cos(2 * phi - 3 * theta)
        + (14771295 / 57278464) * np.cos(2 * phi - theta)
        - 89137125 / 229113856 * np.cos(2 * phi + theta)
        + (48592467 / 114556928) * np.cos(2 * phi + 3 * theta)
        + (332812557 / 229113856) * np.cos(2 * phi + 5 * theta)
        + (51648597 / 57278464) * np.cos(4 * phi - 6 * theta)
        + (86080995 / 57278464) * np.cos(4 * phi - 4 * theta)
        - 34999965 / 28639232 * np.cos(4 * phi - 2 * theta)
        - 59594535 / 57278464 * np.cos(4 * phi + 2 * theta)
        + (131107977 / 57278464) * np.cos(4 * phi + 4 * theta)
        + (258242985 / 114556928) * np.cos(6 * phi - 5 * theta)
        - 36891855 / 57278464 * np.cos(6 * phi - 3 * theta)
        + (218513295 / 229113856) * np.cos(6 * phi - theta)
        - 171972801 / 114556928 * np.cos(6 * phi + theta)
        + (655539885 / 229113856) * np.cos(6 * phi + 3 * theta)
        + (209053845 / 57278464) * np.cos(8 * phi - 4 * theta)
        - 209053845 / 114556928 * np.cos(8 * phi - 2 * theta)
        + (318405087 / 114556928) * np.cos(8 * phi + 2 * theta)
        + (41810769 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (125432307 / 30146560) * np.cos(10 * phi - theta)
        + (106135029 / 60293120) * np.cos(10 * phi + theta)
        + 32089365 / 57278464
    )


def M_gg_1_6_12_0():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_12_0(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (42567525 / 114556928) * np.cos(2 * phi - 3 * theta)
        + (212837625 / 114556928) * np.cos(2 * phi - theta)
        + (127702575 / 57278464) * np.cos(2 * phi + theta)
        + (42567525 / 57278464) * np.cos(2 * phi + 3 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi + 5 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi - 2 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi + 2 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi + 4 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi + 6 * theta)
        + (172297125 / 114556928) * np.cos(6 * phi - theta)
        + (172297125 / 57278464) * np.cos(6 * phi + theta)
        + (73841625 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi + 5 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi + 2 * theta)
        + (14768325 / 24117248) * np.cos(8 * phi + 4 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi + theta)
        + (34459425 / 12058624) * np.cos(10 * phi + 3 * theta)
        + (34459425 / 2097152) * np.cos(12 * phi + 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_1_6_12_1():
    def func(k):
        return (8 / 173918745) * k**12

    return func


def N_gg_1_6_12_1(theta, phi):
    return (
        (10145260125 / 3894935552) * np.cos(4 * phi)
        + (11594583 / 3014656) * np.cos(8 * phi)
        + (9018009 / 524288) * np.cos(12 * phi)
        + (5150670525 / 3894935552) * np.cos(2 * theta)
        + (206026821 / 973733888) * np.cos(4 * theta)
        + (22891869 / 3894935552) * np.cos(6 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi - 5 * theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi - 3 * theta)
        + (2029052025 / 973733888) * np.cos(2 * phi - theta)
        + (2029052025 / 973733888) * np.cos(2 * phi + theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi + 3 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi + 5 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi - 4 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi - 2 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi + 2 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi + 4 * theta)
        + (32207175 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (289864575 / 114556928) * np.cos(6 * phi - theta)
        + (289864575 / 114556928) * np.cos(6 * phi + theta)
        + (32207175 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (27054027 / 6029312) * np.cos(10 * phi - theta)
        + (27054027 / 6029312) * np.cos(10 * phi + theta)
        + 572296725 / 486866944
    )


def M_gg_1_6_12_2():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_12_2(theta, phi):
    return (
        -307432125 / 229113856 * np.cos(4 * phi)
        - 1990989 / 1507328 * np.cos(8 * phi)
        + (2433431 / 524288) * np.cos(12 * phi)
        + (225450225 / 229113856) * np.cos(2 * theta)
        + (9018009 / 7159808) * np.cos(4 * theta)
        + (33066033 / 229113856) * np.cos(6 * theta)
        + (99198099 / 229113856) * np.cos(2 * phi - 5 * theta)
        + (12297285 / 7159808) * np.cos(2 * phi - 3 * theta)
        - 61486425 / 229113856 * np.cos(2 * phi - theta)
        - 20495475 / 28639232 * np.cos(2 * phi + theta)
        + (381215835 / 229113856) * np.cos(2 * phi + 3 * theta)
        + (18621603 / 28639232) * np.cos(2 * phi + 5 * theta)
        + (6441435 / 229113856) * np.cos(2 * phi + 7 * theta)
        + (225450225 / 229113856) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (20495475 / 28639232) * np.cos(4 * phi + 2 * theta)
        + (342567225 / 229113856) * np.cos(4 * phi + 4 * theta)
        + (43918875 / 229113856) * np.cos(4 * phi + 6 * theta)
        + (425850425 / 229113856) * np.cos(6 * phi - 3 * theta)
        + (16591575 / 28639232) * np.cos(6 * phi - theta)
        - 282056775 / 229113856 * np.cos(6 * phi + theta)
        + (57675475 / 28639232) * np.cos(6 * phi + 3 * theta)
        + (82957875 / 114556928) * np.cos(6 * phi + 5 * theta)
        + (36501465 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (9954945 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (11851125 / 6029312) * np.cos(8 * phi + 4 * theta)
        + (51102051 / 12058624) * np.cos(10 * phi - theta)
        - 1990989 / 753664 * np.cos(10 * phi + theta)
        + (49774725 / 12058624) * np.cos(10 * phi + 3 * theta)
        + (3318315 / 524288) * np.cos(12 * phi + 2 * theta)
        - 75150075 / 114556928
    )


def M_gg_1_6_12_3():
    def func(k):
        return (4 / 227432205) * k**12

    return func


def N_gg_1_6_12_3(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi - 5 * theta)
        + (42567525 / 57278464) * np.cos(2 * phi - 3 * theta)
        + (127702575 / 57278464) * np.cos(2 * phi - theta)
        + (212837625 / 114556928) * np.cos(2 * phi + theta)
        + (42567525 / 114556928) * np.cos(2 * phi + 3 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi - 6 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi - 4 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi + 2 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi - 5 * theta)
        + (73841625 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (172297125 / 57278464) * np.cos(6 * phi - theta)
        + (172297125 / 114556928) * np.cos(6 * phi + theta)
        + (14768325 / 24117248) * np.cos(8 * phi - 4 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi - 2 * theta)
        + (34459425 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi - theta)
        + (34459425 / 2097152) * np.cos(12 * phi - 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_1_6_12_4():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_12_4(theta, phi):
    return (
        -307432125 / 229113856 * np.cos(4 * phi)
        - 1990989 / 1507328 * np.cos(8 * phi)
        + (2433431 / 524288) * np.cos(12 * phi)
        + (225450225 / 229113856) * np.cos(2 * theta)
        + (9018009 / 7159808) * np.cos(4 * theta)
        + (33066033 / 229113856) * np.cos(6 * theta)
        + (6441435 / 229113856) * np.cos(2 * phi - 7 * theta)
        + (18621603 / 28639232) * np.cos(2 * phi - 5 * theta)
        + (381215835 / 229113856) * np.cos(2 * phi - 3 * theta)
        - 20495475 / 28639232 * np.cos(2 * phi - theta)
        - 61486425 / 229113856 * np.cos(2 * phi + theta)
        + (12297285 / 7159808) * np.cos(2 * phi + 3 * theta)
        + (99198099 / 229113856) * np.cos(2 * phi + 5 * theta)
        + (43918875 / 229113856) * np.cos(4 * phi - 6 * theta)
        + (342567225 / 229113856) * np.cos(4 * phi - 4 * theta)
        + (20495475 / 28639232) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 114556928) * np.cos(4 * phi + 2 * theta)
        + (225450225 / 229113856) * np.cos(4 * phi + 4 * theta)
        + (82957875 / 114556928) * np.cos(6 * phi - 5 * theta)
        + (57675475 / 28639232) * np.cos(6 * phi - 3 * theta)
        - 282056775 / 229113856 * np.cos(6 * phi - theta)
        + (16591575 / 28639232) * np.cos(6 * phi + theta)
        + (425850425 / 229113856) * np.cos(6 * phi + 3 * theta)
        + (11851125 / 6029312) * np.cos(8 * phi - 4 * theta)
        + (9954945 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (36501465 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (49774725 / 12058624) * np.cos(10 * phi - 3 * theta)
        - 1990989 / 753664 * np.cos(10 * phi - theta)
        + (51102051 / 12058624) * np.cos(10 * phi + theta)
        + (3318315 / 524288) * np.cos(12 * phi - 2 * theta)
        - 75150075 / 114556928
    )


def M_gg_1_6_14_0():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_14_0(theta, phi):
    return (
        (2299592295 / 916455424) * np.cos(4 * phi)
        + (153306153 / 48234496) * np.cos(8 * phi)
        + (12167155 / 2097152) * np.cos(12 * phi)
        + (1289575287 / 916455424) * np.cos(2 * theta)
        + (1289575287 / 4582277120) * np.cos(4 * theta)
        + (61408347 / 4582277120) * np.cos(6 * theta)
        + (99198099 / 1832910848) * np.cos(2 * phi - 5 * theta)
        + (297594297 / 458227712) * np.cos(2 * phi - 3 * theta)
        + (7439857425 / 3665821696) * np.cos(2 * phi - theta)
        + (495990495 / 229113856) * np.cos(2 * phi + theta)
        + (1487971485 / 1832910848) * np.cos(2 * phi + 3 * theta)
        + (42513471 / 458227712) * np.cos(2 * phi + 5 * theta)
        + (14171157 / 7331643392) * np.cos(2 * phi + 7 * theta)
        + (153306153 / 916455424) * np.cos(4 * phi - 4 * theta)
        + (2299592295 / 1832910848) * np.cos(4 * phi - 2 * theta)
        + (766530765 / 458227712) * np.cos(4 * phi + 2 * theta)
        + (328513185 / 916455424) * np.cos(4 * phi + 4 * theta)
        + (65702637 / 3665821696) * np.cos(4 * phi + 6 * theta)
        + (85170085 / 192937984) * np.cos(6 * phi - 3 * theta)
        + (51102051 / 24117248) * np.cos(6 * phi - theta)
        + (255510255 / 96468992) * np.cos(6 * phi + theta)
        + (12167155 / 12058624) * np.cos(6 * phi + 3 * theta)
        + (36501465 / 385875968) * np.cos(6 * phi + 5 * theta)
        + (51102051 / 48234496) * np.cos(8 * phi - 2 * theta)
        + (109504395 / 48234496) * np.cos(8 * phi + 2 * theta)
        + (36501465 / 96468992) * np.cos(8 * phi + 4 * theta)
        + (51102051 / 20971520) * np.cos(10 * phi - theta)
        + (21900879 / 5242880) * np.cos(10 * phi + theta)
        + (21900879 / 16777216) * np.cos(10 * phi + 3 * theta)
        + (36501465 / 8388608) * np.cos(12 * phi + 2 * theta)
        + (328513185 / 16777216) * np.cos(14 * phi + theta)
        + 2149292145 / 1832910848
    )


def M_gg_1_6_14_1():
    def func(k):
        return (8 / 2956618665) * k**12

    return func


def N_gg_1_6_14_1(theta, phi):
    return (
        (2299592295 / 916455424) * np.cos(4 * phi)
        + (153306153 / 48234496) * np.cos(8 * phi)
        + (12167155 / 2097152) * np.cos(12 * phi)
        + (1289575287 / 916455424) * np.cos(2 * theta)
        + (1289575287 / 4582277120) * np.cos(4 * theta)
        + (61408347 / 4582277120) * np.cos(6 * theta)
        + (14171157 / 7331643392) * np.cos(2 * phi - 7 * theta)
        + (42513471 / 458227712) * np.cos(2 * phi - 5 * theta)
        + (1487971485 / 1832910848) * np.cos(2 * phi - 3 * theta)
        + (495990495 / 229113856) * np.cos(2 * phi - theta)
        + (7439857425 / 3665821696) * np.cos(2 * phi + theta)
        + (297594297 / 458227712) * np.cos(2 * phi + 3 * theta)
        + (99198099 / 1832910848) * np.cos(2 * phi + 5 * theta)
        + (65702637 / 3665821696) * np.cos(4 * phi - 6 * theta)
        + (328513185 / 916455424) * np.cos(4 * phi - 4 * theta)
        + (766530765 / 458227712) * np.cos(4 * phi - 2 * theta)
        + (2299592295 / 1832910848) * np.cos(4 * phi + 2 * theta)
        + (153306153 / 916455424) * np.cos(4 * phi + 4 * theta)
        + (36501465 / 385875968) * np.cos(6 * phi - 5 * theta)
        + (12167155 / 12058624) * np.cos(6 * phi - 3 * theta)
        + (255510255 / 96468992) * np.cos(6 * phi - theta)
        + (51102051 / 24117248) * np.cos(6 * phi + theta)
        + (85170085 / 192937984) * np.cos(6 * phi + 3 * theta)
        + (36501465 / 96468992) * np.cos(8 * phi - 4 * theta)
        + (109504395 / 48234496) * np.cos(8 * phi - 2 * theta)
        + (51102051 / 48234496) * np.cos(8 * phi + 2 * theta)
        + (21900879 / 16777216) * np.cos(10 * phi - 3 * theta)
        + (21900879 / 5242880) * np.cos(10 * phi - theta)
        + (51102051 / 20971520) * np.cos(10 * phi + theta)
        + (36501465 / 8388608) * np.cos(12 * phi - 2 * theta)
        + (328513185 / 16777216) * np.cos(14 * phi - theta)
        + 2149292145 / 1832910848
    )


def M_gg_2_0_0_0():
    def func(k):
        return 1 / 9

    return func


def N_gg_2_0_0_0(theta, phi):
    return 1


def M_gg_2_0_0_1():
    def func(k):
        return 4 / 225

    return func


def N_gg_2_0_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_0_2_0():
    def func(k):
        return 2 / 45

    return func


def N_gg_2_0_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_2_0_2_1():
    def func(k):
        return 2 / 45

    return func


def N_gg_2_0_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_2_0_2_2():
    def func(k):
        return 4 / 225

    return func


def N_gg_2_0_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_2_0_4_0():
    def func(k):
        return 4 / 225

    return func


def N_gg_2_0_4_0(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_1_0_0():
    def func(k):
        return -1 / 15 * k**2

    return func


def N_gg_2_1_0_0(theta, phi):
    return 1


def M_gg_2_1_0_1():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_2_1_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_1_2_0():
    def func(k):
        return -17 / 525 * k**2

    return func


def N_gg_2_1_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_2_1_2_1():
    def func(k):
        return -17 / 525 * k**2

    return func


def N_gg_2_1_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_2_1_2_2():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_2_1_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_2_1_2_3():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_2_1_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_2_1_2_4():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_2_1_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_1_4_0():
    def func(k):
        return -4 / 945 * k**2

    return func


def N_gg_2_1_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_2_1_4_1():
    def func(k):
        return -8 / 525 * k**2

    return func


def N_gg_2_1_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_1_4_2():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_2_1_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_1_4_3():
    def func(k):
        return -4 / 945 * k**2

    return func


def N_gg_2_1_4_3(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_2_1_4_4():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_2_1_4_4(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_1_6_0():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_2_1_6_0(theta, phi):
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


def M_gg_2_1_6_1():
    def func(k):
        return -8 / 4725 * k**2

    return func


def N_gg_2_1_6_1(theta, phi):
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


def M_gg_2_2_0_0():
    def func(k):
        return (23 / 1050) * k**4

    return func


def N_gg_2_2_0_0(theta, phi):
    return 1


def M_gg_2_2_0_1():
    def func(k):
        return (71 / 11025) * k**4

    return func


def N_gg_2_2_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_2_0_2():
    def func(k):
        return (16 / 99225) * k**4

    return func


def N_gg_2_2_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_2_2_2_0():
    def func(k):
        return (19 / 1575) * k**4

    return func


def N_gg_2_2_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_2_2_2_1():
    def func(k):
        return (19 / 1575) * k**4

    return func


def N_gg_2_2_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_2_2_2_2():
    def func(k):
        return (71 / 11025) * k**4

    return func


def N_gg_2_2_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_2_2_2_3():
    def func(k):
        return (158 / 121275) * k**4

    return func


def N_gg_2_2_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_2_2_2_4():
    def func(k):
        return (158 / 121275) * k**4

    return func


def N_gg_2_2_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_2_2_5():
    def func(k):
        return (16 / 99225) * k**4

    return func


def N_gg_2_2_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_2_2_4_0():
    def func(k):
        return (47 / 17325) * k**4

    return func


def N_gg_2_2_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_2_2_4_1():
    def func(k):
        return (71 / 11025) * k**4

    return func


def N_gg_2_2_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_2_4_2():
    def func(k):
        return (158 / 121275) * k**4

    return func


def N_gg_2_2_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_2_4_3():
    def func(k):
        return (4 / 45045) * k**4

    return func


def N_gg_2_2_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_2_2_4_4():
    def func(k):
        return (47 / 17325) * k**4

    return func


def N_gg_2_2_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_2_2_4_5():
    def func(k):
        return (158 / 121275) * k**4

    return func


def N_gg_2_2_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_2_4_6():
    def func(k):
        return (16 / 99225) * k**4

    return func


def N_gg_2_2_4_6(theta, phi):
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


def M_gg_2_2_4_7():
    def func(k):
        return (4 / 45045) * k**4

    return func


def N_gg_2_2_4_7(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_2_2_6_0():
    def func(k):
        return (2 / 9009) * k**4

    return func


def N_gg_2_2_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_2_2_6_1():
    def func(k):
        return (158 / 121275) * k**4

    return func


def N_gg_2_2_6_1(theta, phi):
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


def M_gg_2_2_6_2():
    def func(k):
        return (4 / 45045) * k**4

    return func


def N_gg_2_2_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_2_2_6_3():
    def func(k):
        return (158 / 121275) * k**4

    return func


def N_gg_2_2_6_3(theta, phi):
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


def M_gg_2_2_6_4():
    def func(k):
        return (16 / 99225) * k**4

    return func


def N_gg_2_2_6_4(theta, phi):
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


def M_gg_2_2_6_5():
    def func(k):
        return (2 / 9009) * k**4

    return func


def N_gg_2_2_6_5(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_2_2_6_6():
    def func(k):
        return (4 / 45045) * k**4

    return func


def N_gg_2_2_6_6(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_2_2_8_0():
    def func(k):
        return (4 / 45045) * k**4

    return func


def N_gg_2_2_8_0(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_2_8_1():
    def func(k):
        return (16 / 99225) * k**4

    return func


def N_gg_2_2_8_1(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_2_2_8_2():
    def func(k):
        return (4 / 45045) * k**4

    return func


def N_gg_2_2_8_2(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_3_0_0():
    def func(k):
        return -29 / 5670 * k**6

    return func


def N_gg_2_3_0_0(theta, phi):
    return 1


def M_gg_2_3_0_1():
    def func(k):
        return -79 / 43659 * k**6

    return func


def N_gg_2_3_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_3_0_2():
    def func(k):
        return -8 / 72765 * k**6

    return func


def N_gg_2_3_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_2_3_2_0():
    def func(k):
        return -269 / 87318 * k**6

    return func


def N_gg_2_3_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_2_3_2_1():
    def func(k):
        return -269 / 87318 * k**6

    return func


def N_gg_2_3_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_2_3_2_2():
    def func(k):
        return -79 / 43659 * k**6

    return func


def N_gg_2_3_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_2_3_2_3():
    def func(k):
        return -95 / 189189 * k**6

    return func


def N_gg_2_3_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_2_3_2_4():
    def func(k):
        return -95 / 189189 * k**6

    return func


def N_gg_2_3_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_3_2_5():
    def func(k):
        return -8 / 72765 * k**6

    return func


def N_gg_2_3_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_2_3_2_6():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_2_3_2_7():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_2_3_4_0():
    def func(k):
        return -79 / 85995 * k**6

    return func


def N_gg_2_3_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_2_3_4_1():
    def func(k):
        return -79 / 43659 * k**6

    return func


def N_gg_2_3_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_3_4_2():
    def func(k):
        return -95 / 189189 * k**6

    return func


def N_gg_2_3_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_3_4_3():
    def func(k):
        return -932 / 14189175 * k**6

    return func


def N_gg_2_3_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_2_3_4_4():
    def func(k):
        return -79 / 85995 * k**6

    return func


def N_gg_2_3_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_2_3_4_5():
    def func(k):
        return -95 / 189189 * k**6

    return func


def N_gg_2_3_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_3_4_6():
    def func(k):
        return -8 / 72765 * k**6

    return func


def N_gg_2_3_4_6(theta, phi):
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


def M_gg_2_3_4_7():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_2_3_4_8():
    def func(k):
        return -932 / 14189175 * k**6

    return func


def N_gg_2_3_4_8(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_2_3_4_9():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_4_9(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_2_3_6_0():
    def func(k):
        return -1 / 7371 * k**6

    return func


def N_gg_2_3_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_2_3_6_1():
    def func(k):
        return -95 / 189189 * k**6

    return func


def N_gg_2_3_6_1(theta, phi):
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


def M_gg_2_3_6_2():
    def func(k):
        return -932 / 14189175 * k**6

    return func


def N_gg_2_3_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_2_3_6_3():
    def func(k):
        return -16 / 4922775 * k**6

    return func


def N_gg_2_3_6_3(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (9009 / 2048) * np.cos(4 * phi + 4 * theta)
        + (693 / 4096) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(6 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_2_3_6_4():
    def func(k):
        return -95 / 189189 * k**6

    return func


def N_gg_2_3_6_4(theta, phi):
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


def M_gg_2_3_6_5():
    def func(k):
        return -8 / 72765 * k**6

    return func


def N_gg_2_3_6_5(theta, phi):
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


def M_gg_2_3_6_6():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_6_6(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_2_3_6_7():
    def func(k):
        return -1 / 7371 * k**6

    return func


def N_gg_2_3_6_7(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_2_3_6_8():
    def func(k):
        return -932 / 14189175 * k**6

    return func


def N_gg_2_3_6_8(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_2_3_6_9():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_6_9(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_2_3_6_10():
    def func(k):
        return -16 / 4922775 * k**6

    return func


def N_gg_2_3_6_10(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (9009 / 2048) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (45045 / 4096) * np.cos(6 * phi - 5 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (693 / 4096) * np.cos(6 * phi - theta)
        + 1225 / 1024
    )


def M_gg_2_3_8_0():
    def func(k):
        return -8 / 984555 * k**6

    return func


def N_gg_2_3_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi + theta)
        + (11781 / 4096) * np.cos(4 * phi + 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi + 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi + 4 * theta)
        + 20825 / 16384
    )


def M_gg_2_3_8_1():
    def func(k):
        return -932 / 14189175 * k**6

    return func


def N_gg_2_3_8_1(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_3_8_2():
    def func(k):
        return -16 / 4922775 * k**6

    return func


def N_gg_2_3_8_2(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (187425 / 77824) * np.cos(2 * phi - theta)
        - 26775 / 19456 * np.cos(2 * phi + theta)
        + (176715 / 77824) * np.cos(2 * phi + 3 * theta)
        - 11781 / 19456 * np.cos(4 * phi + 2 * theta)
        + (153153 / 77824) * np.cos(4 * phi + 4 * theta)
        + (153153 / 77824) * np.cos(6 * phi + theta)
        + (21879 / 19456) * np.cos(6 * phi + 3 * theta)
        + (109395 / 77824) * np.cos(6 * phi + 5 * theta)
        + (109395 / 77824) * np.cos(8 * phi + 2 * theta)
        + (546975 / 77824) * np.cos(8 * phi + 4 * theta)
        - 62475 / 77824
    )


def M_gg_2_3_8_3():
    def func(k):
        return -8 / 72765 * k**6

    return func


def N_gg_2_3_8_3(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_2_3_8_4():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_2_3_8_5():
    def func(k):
        return -932 / 14189175 * k**6

    return func


def N_gg_2_3_8_5(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_3_8_6():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_8_6(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_2_3_8_7():
    def func(k):
        return -8 / 984555 * k**6

    return func


def N_gg_2_3_8_7(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi - theta)
        + (11781 / 4096) * np.cos(4 * phi - 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi - 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi - 4 * theta)
        + 20825 / 16384
    )


def M_gg_2_3_8_8():
    def func(k):
        return -16 / 4922775 * k**6

    return func


def N_gg_2_3_8_8(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (176715 / 77824) * np.cos(2 * phi - 3 * theta)
        - 26775 / 19456 * np.cos(2 * phi - theta)
        + (187425 / 77824) * np.cos(2 * phi + theta)
        + (153153 / 77824) * np.cos(4 * phi - 4 * theta)
        - 11781 / 19456 * np.cos(4 * phi - 2 * theta)
        + (109395 / 77824) * np.cos(6 * phi - 5 * theta)
        + (21879 / 19456) * np.cos(6 * phi - 3 * theta)
        + (153153 / 77824) * np.cos(6 * phi - theta)
        + (546975 / 77824) * np.cos(8 * phi - 4 * theta)
        + (109395 / 77824) * np.cos(8 * phi - 2 * theta)
        - 62475 / 77824
    )


def M_gg_2_3_10_0():
    def func(k):
        return -16 / 4922775 * k**6

    return func


def N_gg_2_3_10_0(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (1819125 / 1245184) * np.cos(2 * phi - theta)
        + (363825 / 155648) * np.cos(2 * phi + theta)
        + (363825 / 622592) * np.cos(2 * phi + 3 * theta)
        + (675675 / 311296) * np.cos(4 * phi + 2 * theta)
        + (96525 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 622592) * np.cos(6 * phi + theta)
        + (289575 / 155648) * np.cos(6 * phi + 3 * theta)
        + (289575 / 2490368) * np.cos(6 * phi + 5 * theta)
        + (1640925 / 311296) * np.cos(8 * phi + 2 * theta)
        + (1640925 / 1245184) * np.cos(8 * phi + 4 * theta)
        + (1640925 / 131072) * np.cos(10 * phi + 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_2_3_10_1():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_2_3_10_2():
    def func(k):
        return -8 / 945945 * k**6

    return func


def N_gg_2_3_10_2(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_2_3_10_3():
    def func(k):
        return -16 / 4922775 * k**6

    return func


def N_gg_2_3_10_3(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (363825 / 622592) * np.cos(2 * phi - 3 * theta)
        + (363825 / 155648) * np.cos(2 * phi - theta)
        + (1819125 / 1245184) * np.cos(2 * phi + theta)
        + (96525 / 311296) * np.cos(4 * phi - 4 * theta)
        + (675675 / 311296) * np.cos(4 * phi - 2 * theta)
        + (289575 / 2490368) * np.cos(6 * phi - 5 * theta)
        + (289575 / 155648) * np.cos(6 * phi - 3 * theta)
        + (2027025 / 622592) * np.cos(6 * phi - theta)
        + (1640925 / 1245184) * np.cos(8 * phi - 4 * theta)
        + (1640925 / 311296) * np.cos(8 * phi - 2 * theta)
        + (1640925 / 131072) * np.cos(10 * phi - 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_2_4_0_0():
    def func(k):
        return (331 / 423360) * k**8

    return func


def N_gg_2_4_0_0(theta, phi):
    return 1


def M_gg_2_4_0_1():
    def func(k):
        return (389 / 1164240) * k**8

    return func


def N_gg_2_4_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_4_0_2():
    def func(k):
        return (1201 / 31216185) * k**8

    return func


def N_gg_2_4_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_2_4_0_3():
    def func(k):
        return (4 / 9018009) * k**8

    return func


def N_gg_2_4_0_3(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * theta)
        + (819 / 256) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + 325 / 256
    )


def M_gg_2_4_2_0():
    def func(k):
        return (239 / 465696) * k**8

    return func


def N_gg_2_4_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_2_4_2_1():
    def func(k):
        return (239 / 465696) * k**8

    return func


def N_gg_2_4_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_2_4_2_2():
    def func(k):
        return (389 / 1164240) * k**8

    return func


def N_gg_2_4_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_2_4_2_3():
    def func(k):
        return (799 / 6810804) * k**8

    return func


def N_gg_2_4_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_2_4_2_4():
    def func(k):
        return (799 / 6810804) * k**8

    return func


def N_gg_2_4_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_4_2_5():
    def func(k):
        return (1201 / 31216185) * k**8

    return func


def N_gg_2_4_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_2_4_2_6():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_2_4_2_7():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_2_4_2_8():
    def func(k):
        return (4 / 9018009) * k**8

    return func


def N_gg_2_4_2_8(theta, phi):
    return (
        -6825 / 5632 * np.cos(2 * theta)
        + (819 / 2816) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + (819 / 512) * np.cos(2 * phi - 5 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 512) * np.cos(2 * phi + 5 * theta)
        - 2275 / 2816
    )


def M_gg_2_4_4_0():
    def func(k):
        return (12559 / 68108040) * k**8

    return func


def N_gg_2_4_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_2_4_4_1():
    def func(k):
        return (389 / 1164240) * k**8

    return func


def N_gg_2_4_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_4_4_2():
    def func(k):
        return (799 / 6810804) * k**8

    return func


def N_gg_2_4_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_4_4_3():
    def func(k):
        return (17 / 859950) * k**8

    return func


def N_gg_2_4_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_2_4_4_4():
    def func(k):
        return (12559 / 68108040) * k**8

    return func


def N_gg_2_4_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_2_4_4_5():
    def func(k):
        return (799 / 6810804) * k**8

    return func


def N_gg_2_4_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_4_4_6():
    def func(k):
        return (1201 / 31216185) * k**8

    return func


def N_gg_2_4_4_6(theta, phi):
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


def M_gg_2_4_4_7():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_2_4_4_8():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_4_8(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (77175 / 73216) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (19845 / 6656) * np.cos(2 * phi + 3 * theta)
        + (6615 / 2048) * np.cos(2 * phi + 5 * theta)
        + (77175 / 1171456) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(4 * phi + 4 * theta)
        + (99225 / 8192) * np.cos(4 * phi + 6 * theta)
        + 694575 / 585728
    )


def M_gg_2_4_4_9():
    def func(k):
        return (17 / 859950) * k**8

    return func


def N_gg_2_4_4_9(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_2_4_4_10():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_4_10(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_2_4_4_11():
    def func(k):
        return (4 / 9018009) * k**8

    return func


def N_gg_2_4_4_11(theta, phi):
    return (
        (1003275 / 382976) * np.cos(4 * phi)
        + (12285 / 69632) * np.cos(2 * theta)
        - 22113 / 11968 * np.cos(4 * theta)
        + (243243 / 69632) * np.cos(6 * theta)
        + (110565 / 34816) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(2 * phi + 5 * theta)
        + (85995 / 69632) * np.cos(4 * phi - 4 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (85995 / 69632) * np.cos(4 * phi + 4 * theta)
        + 61425 / 95744
    )


def M_gg_2_4_4_12():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_4_12(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (6615 / 2048) * np.cos(2 * phi - 5 * theta)
        + (19845 / 6656) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (77175 / 73216) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (99225 / 8192) * np.cos(4 * phi - 6 * theta)
        + (6615 / 2048) * np.cos(4 * phi - 4 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (77175 / 1171456) * np.cos(4 * phi + 2 * theta)
        + 694575 / 585728
    )


def M_gg_2_4_6_0():
    def func(k):
        return (617 / 18918900) * k**8

    return func


def N_gg_2_4_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_2_4_6_1():
    def func(k):
        return (799 / 6810804) * k**8

    return func


def N_gg_2_4_6_1(theta, phi):
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


def M_gg_2_4_6_2():
    def func(k):
        return (17 / 859950) * k**8

    return func


def N_gg_2_4_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_2_4_6_3():
    def func(k):
        return (16 / 11486475) * k**8

    return func


def N_gg_2_4_6_3(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (9009 / 2048) * np.cos(4 * phi + 4 * theta)
        + (693 / 4096) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(6 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_2_4_6_4():
    def func(k):
        return (799 / 6810804) * k**8

    return func


def N_gg_2_4_6_4(theta, phi):
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


def M_gg_2_4_6_5():
    def func(k):
        return (1201 / 31216185) * k**8

    return func


def N_gg_2_4_6_5(theta, phi):
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


def M_gg_2_4_6_6():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_6_6(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_2_4_6_7():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_6_7(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (1289925 / 856064) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        - 104895 / 77824 * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(2 * phi + 5 * theta)
        + (694575 / 856064) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        - 36855 / 19456 * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 6 * theta)
        + (19845 / 77824) * np.cos(6 * phi - theta)
        + (93555 / 77824) * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(6 * phi + 5 * theta)
        - 297675 / 428032
    )


def M_gg_2_4_6_8():
    def func(k):
        return (617 / 18918900) * k**8

    return func


def N_gg_2_4_6_8(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_2_4_6_9():
    def func(k):
        return (17 / 859950) * k**8

    return func


def N_gg_2_4_6_9(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_2_4_6_10():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_6_10(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_2_4_6_11():
    def func(k):
        return (4 / 9018009) * k**8

    return func


def N_gg_2_4_6_11(theta, phi):
    return (
        -5589675 / 3638272 * np.cos(4 * phi)
        + (443625 / 661504) * np.cos(2 * theta)
        + (266175 / 909568) * np.cos(4 * theta)
        + (975975 / 661504) * np.cos(6 * theta)
        + (1863225 / 661504) * np.cos(2 * phi - 5 * theta)
        - 5589675 / 3638272 * np.cos(2 * phi - 3 * theta)
        + (443625 / 661504) * np.cos(2 * phi - theta)
        + (443625 / 661504) * np.cos(2 * phi + theta)
        - 5589675 / 3638272 * np.cos(2 * phi + 3 * theta)
        + (1863225 / 661504) * np.cos(2 * phi + 5 * theta)
        + (1863225 / 661504) * np.cos(4 * phi - 4 * theta)
        + (266175 / 909568) * np.cos(4 * phi - 2 * theta)
        + (266175 / 909568) * np.cos(4 * phi + 2 * theta)
        + (1863225 / 661504) * np.cos(4 * phi + 4 * theta)
        + (975975 / 661504) * np.cos(6 * phi - 3 * theta)
        + (1863225 / 661504) * np.cos(6 * phi - theta)
        + (1863225 / 661504) * np.cos(6 * phi + theta)
        + (975975 / 661504) * np.cos(6 * phi + 3 * theta)
        - 528125 / 909568
    )


def M_gg_2_4_6_12():
    def func(k):
        return (16 / 11486475) * k**8

    return func


def N_gg_2_4_6_12(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (9009 / 2048) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (45045 / 4096) * np.cos(6 * phi - 5 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (693 / 4096) * np.cos(6 * phi - theta)
        + 1225 / 1024
    )


def M_gg_2_4_6_13():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_6_13(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (257985 / 77824) * np.cos(2 * phi - 5 * theta)
        - 104895 / 77824 * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (1289925 / 856064) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 6 * theta)
        - 36855 / 19456 * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (694575 / 856064) * np.cos(4 * phi + 2 * theta)
        + (405405 / 77824) * np.cos(6 * phi - 5 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        + (93555 / 77824) * np.cos(6 * phi - theta)
        + (19845 / 77824) * np.cos(6 * phi + theta)
        - 297675 / 428032
    )


def M_gg_2_4_8_0():
    def func(k):
        return (4 / 1640925) * k**8

    return func


def N_gg_2_4_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi + theta)
        + (11781 / 4096) * np.cos(4 * phi + 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi + 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi + 4 * theta)
        + 20825 / 16384
    )


def M_gg_2_4_8_1():
    def func(k):
        return (17 / 859950) * k**8

    return func


def N_gg_2_4_8_1(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_4_8_2():
    def func(k):
        return (16 / 11486475) * k**8

    return func


def N_gg_2_4_8_2(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (187425 / 77824) * np.cos(2 * phi - theta)
        - 26775 / 19456 * np.cos(2 * phi + theta)
        + (176715 / 77824) * np.cos(2 * phi + 3 * theta)
        - 11781 / 19456 * np.cos(4 * phi + 2 * theta)
        + (153153 / 77824) * np.cos(4 * phi + 4 * theta)
        + (153153 / 77824) * np.cos(6 * phi + theta)
        + (21879 / 19456) * np.cos(6 * phi + 3 * theta)
        + (109395 / 77824) * np.cos(6 * phi + 5 * theta)
        + (109395 / 77824) * np.cos(8 * phi + 2 * theta)
        + (546975 / 77824) * np.cos(8 * phi + 4 * theta)
        - 62475 / 77824
    )


def M_gg_2_4_8_3():
    def func(k):
        return (1201 / 31216185) * k**8

    return func


def N_gg_2_4_8_3(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_2_4_8_4():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_2_4_8_5():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_8_5(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (5060475 / 2023424) * np.cos(2 * phi - 3 * theta)
        - 5060475 / 4046848 * np.cos(2 * phi - theta)
        + (2457945 / 4046848) * np.cos(2 * phi + theta)
        - 530145 / 4046848 * np.cos(2 * phi + 3 * theta)
        + (530145 / 311296) * np.cos(2 * phi + 5 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi - 2 * theta)
        - 954261 / 1011712 * np.cos(4 * phi + 2 * theta)
        + (530145 / 311296) * np.cos(4 * phi + 4 * theta)
        + (530145 / 622592) * np.cos(4 * phi + 6 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        - 590733 / 311296 * np.cos(6 * phi + 3 * theta)
        + (984555 / 311296) * np.cos(6 * phi + 5 * theta)
        + (984555 / 311296) * np.cos(8 * phi + 2 * theta)
        + (2953665 / 622592) * np.cos(8 * phi + 4 * theta)
        + 5060475 / 8093696
    )


def M_gg_2_4_8_6():
    def func(k):
        return (17 / 859950) * k**8

    return func


def N_gg_2_4_8_6(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_4_8_7():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_8_7(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_2_4_8_8():
    def func(k):
        return (4 / 9018009) * k**8

    return func


def N_gg_2_4_8_8(theta, phi):
    return (
        (429975 / 311296) * np.cos(4 * phi)
        + (2395575 / 622592) * np.cos(8 * phi)
        - 16960125 / 13697024 * np.cos(2 * theta)
        + (12755925 / 6848512) * np.cos(4 * theta)
        + (525525 / 1245184) * np.cos(6 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (429975 / 311296) * np.cos(2 * phi + 5 * theta)
        + (1576575 / 622592) * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        + (1576575 / 622592) * np.cos(4 * phi + 4 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        - 266175 / 311296 * np.cos(6 * phi - theta)
        - 266175 / 311296 * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi - 2 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi + 2 * theta)
        + 1990625 / 3424256
    )


def M_gg_2_4_8_9():
    def func(k):
        return (4 / 1640925) * k**8

    return func


def N_gg_2_4_8_9(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi - theta)
        + (11781 / 4096) * np.cos(4 * phi - 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi - 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi - 4 * theta)
        + 20825 / 16384
    )


def M_gg_2_4_8_10():
    def func(k):
        return (16 / 11486475) * k**8

    return func


def N_gg_2_4_8_10(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (176715 / 77824) * np.cos(2 * phi - 3 * theta)
        - 26775 / 19456 * np.cos(2 * phi - theta)
        + (187425 / 77824) * np.cos(2 * phi + theta)
        + (153153 / 77824) * np.cos(4 * phi - 4 * theta)
        - 11781 / 19456 * np.cos(4 * phi - 2 * theta)
        + (109395 / 77824) * np.cos(6 * phi - 5 * theta)
        + (21879 / 19456) * np.cos(6 * phi - 3 * theta)
        + (153153 / 77824) * np.cos(6 * phi - theta)
        + (546975 / 77824) * np.cos(8 * phi - 4 * theta)
        + (109395 / 77824) * np.cos(8 * phi - 2 * theta)
        - 62475 / 77824
    )


def M_gg_2_4_8_11():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_8_11(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (530145 / 311296) * np.cos(2 * phi - 5 * theta)
        - 530145 / 4046848 * np.cos(2 * phi - 3 * theta)
        + (2457945 / 4046848) * np.cos(2 * phi - theta)
        - 5060475 / 4046848 * np.cos(2 * phi + theta)
        + (5060475 / 2023424) * np.cos(2 * phi + 3 * theta)
        + (530145 / 622592) * np.cos(4 * phi - 6 * theta)
        + (530145 / 311296) * np.cos(4 * phi - 4 * theta)
        - 954261 / 1011712 * np.cos(4 * phi - 2 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi + 2 * theta)
        + (984555 / 311296) * np.cos(6 * phi - 5 * theta)
        - 590733 / 311296 * np.cos(6 * phi - 3 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        + (2953665 / 622592) * np.cos(8 * phi - 4 * theta)
        + (984555 / 311296) * np.cos(8 * phi - 2 * theta)
        + 5060475 / 8093696
    )


def M_gg_2_4_10_0():
    def func(k):
        return (16 / 11486475) * k**8

    return func


def N_gg_2_4_10_0(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (1819125 / 1245184) * np.cos(2 * phi - theta)
        + (363825 / 155648) * np.cos(2 * phi + theta)
        + (363825 / 622592) * np.cos(2 * phi + 3 * theta)
        + (675675 / 311296) * np.cos(4 * phi + 2 * theta)
        + (96525 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 622592) * np.cos(6 * phi + theta)
        + (289575 / 155648) * np.cos(6 * phi + 3 * theta)
        + (289575 / 2490368) * np.cos(6 * phi + 5 * theta)
        + (1640925 / 311296) * np.cos(8 * phi + 2 * theta)
        + (1640925 / 1245184) * np.cos(8 * phi + 4 * theta)
        + (1640925 / 131072) * np.cos(10 * phi + 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_2_4_10_1():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_2_4_10_2():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_10_2(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (22920975 / 14319616) * np.cos(2 * phi - 3 * theta)
        + (363825 / 894976) * np.cos(2 * phi - theta)
        - 3274425 / 3579904 * np.cos(2 * phi + theta)
        + (12879405 / 7159808) * np.cos(2 * phi + 3 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi + 5 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi - 2 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi + 2 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (42567525 / 14319616) * np.cos(6 * phi - theta)
        - 13378365 / 7159808 * np.cos(6 * phi + theta)
        + (7123545 / 3579904) * np.cos(6 * phi + 3 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi + 5 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi + 2 * theta)
        + (44304975 / 14319616) * np.cos(8 * phi + 4 * theta)
        + (2297295 / 753664) * np.cos(10 * phi + theta)
        + (4922775 / 753664) * np.cos(10 * phi + 3 * theta)
        - 9823275 / 14319616
    )


def M_gg_2_4_10_3():
    def func(k):
        return (2582 / 468242775) * k**8

    return func


def N_gg_2_4_10_3(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_2_4_10_4():
    def func(k):
        return (4 / 9018009) * k**8

    return func


def N_gg_2_4_10_4(theta, phi):
    return (
        -184459275 / 121716736 * np.cos(4 * phi)
        - 36891855 / 14319616 * np.cos(8 * phi)
        + (568856925 / 486866944) * np.cos(2 * theta)
        + (269800713 / 243433472) * np.cos(4 * theta)
        + (35756721 / 486866944) * np.cos(6 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi - 5 * theta)
        + (854188335 / 486866944) * np.cos(2 * phi - 3 * theta)
        - 99324225 / 243433472 * np.cos(2 * phi - theta)
        - 99324225 / 243433472 * np.cos(2 * phi + theta)
        + (854188335 / 486866944) * np.cos(2 * phi + 3 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi + 5 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi + 2 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi + 4 * theta)
        + (553377825 / 243433472) * np.cos(6 * phi - 3 * theta)
        - 110675565 / 486866944 * np.cos(6 * phi - theta)
        - 110675565 / 486866944 * np.cos(6 * phi + theta)
        + (553377825 / 243433472) * np.cos(6 * phi + 3 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi - 2 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi + 2 * theta)
        + (7378371 / 1507328) * np.cos(10 * phi - theta)
        + (7378371 / 1507328) * np.cos(10 * phi + theta)
        - 81265275 / 121716736
    )


def M_gg_2_4_10_5():
    def func(k):
        return (16 / 11486475) * k**8

    return func


def N_gg_2_4_10_5(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (363825 / 622592) * np.cos(2 * phi - 3 * theta)
        + (363825 / 155648) * np.cos(2 * phi - theta)
        + (1819125 / 1245184) * np.cos(2 * phi + theta)
        + (96525 / 311296) * np.cos(4 * phi - 4 * theta)
        + (675675 / 311296) * np.cos(4 * phi - 2 * theta)
        + (289575 / 2490368) * np.cos(6 * phi - 5 * theta)
        + (289575 / 155648) * np.cos(6 * phi - 3 * theta)
        + (2027025 / 622592) * np.cos(6 * phi - theta)
        + (1640925 / 1245184) * np.cos(8 * phi - 4 * theta)
        + (1640925 / 311296) * np.cos(8 * phi - 2 * theta)
        + (1640925 / 131072) * np.cos(10 * phi - 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_2_4_10_6():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_10_6(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi - 5 * theta)
        + (12879405 / 7159808) * np.cos(2 * phi - 3 * theta)
        - 3274425 / 3579904 * np.cos(2 * phi - theta)
        + (363825 / 894976) * np.cos(2 * phi + theta)
        + (22920975 / 14319616) * np.cos(2 * phi + 3 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi - 6 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi - 4 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi - 2 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi + 2 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi - 5 * theta)
        + (7123545 / 3579904) * np.cos(6 * phi - 3 * theta)
        - 13378365 / 7159808 * np.cos(6 * phi - theta)
        + (42567525 / 14319616) * np.cos(6 * phi + theta)
        + (44304975 / 14319616) * np.cos(8 * phi - 4 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi - 2 * theta)
        + (4922775 / 753664) * np.cos(10 * phi - 3 * theta)
        + (2297295 / 753664) * np.cos(10 * phi - theta)
        - 9823275 / 14319616
    )


def M_gg_2_4_12_0():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_12_0(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (42567525 / 114556928) * np.cos(2 * phi - 3 * theta)
        + (212837625 / 114556928) * np.cos(2 * phi - theta)
        + (127702575 / 57278464) * np.cos(2 * phi + theta)
        + (42567525 / 57278464) * np.cos(2 * phi + 3 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi + 5 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi - 2 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi + 2 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi + 4 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi + 6 * theta)
        + (172297125 / 114556928) * np.cos(6 * phi - theta)
        + (172297125 / 57278464) * np.cos(6 * phi + theta)
        + (73841625 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi + 5 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi + 2 * theta)
        + (14768325 / 24117248) * np.cos(8 * phi + 4 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi + theta)
        + (34459425 / 12058624) * np.cos(10 * phi + 3 * theta)
        + (34459425 / 2097152) * np.cos(12 * phi + 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_2_4_12_1():
    def func(k):
        return (4 / 9018009) * k**8

    return func


def N_gg_2_4_12_1(theta, phi):
    return (
        (10145260125 / 3894935552) * np.cos(4 * phi)
        + (11594583 / 3014656) * np.cos(8 * phi)
        + (9018009 / 524288) * np.cos(12 * phi)
        + (5150670525 / 3894935552) * np.cos(2 * theta)
        + (206026821 / 973733888) * np.cos(4 * theta)
        + (22891869 / 3894935552) * np.cos(6 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi - 5 * theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi - 3 * theta)
        + (2029052025 / 973733888) * np.cos(2 * phi - theta)
        + (2029052025 / 973733888) * np.cos(2 * phi + theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi + 3 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi + 5 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi - 4 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi - 2 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi + 2 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi + 4 * theta)
        + (32207175 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (289864575 / 114556928) * np.cos(6 * phi - theta)
        + (289864575 / 114556928) * np.cos(6 * phi + theta)
        + (32207175 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (27054027 / 6029312) * np.cos(10 * phi - theta)
        + (27054027 / 6029312) * np.cos(10 * phi + theta)
        + 572296725 / 486866944
    )


def M_gg_2_4_12_2():
    def func(k):
        return (32 / 103378275) * k**8

    return func


def N_gg_2_4_12_2(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi - 5 * theta)
        + (42567525 / 57278464) * np.cos(2 * phi - 3 * theta)
        + (127702575 / 57278464) * np.cos(2 * phi - theta)
        + (212837625 / 114556928) * np.cos(2 * phi + theta)
        + (42567525 / 114556928) * np.cos(2 * phi + 3 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi - 6 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi - 4 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi + 2 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi - 5 * theta)
        + (73841625 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (172297125 / 57278464) * np.cos(6 * phi - theta)
        + (172297125 / 114556928) * np.cos(6 * phi + theta)
        + (14768325 / 24117248) * np.cos(8 * phi - 4 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi - 2 * theta)
        + (34459425 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi - theta)
        + (34459425 / 2097152) * np.cos(12 * phi - 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_2_5_0_0():
    def func(k):
        return -1 / 12096 * k**10

    return func


def N_gg_2_5_0_0(theta, phi):
    return 1


def M_gg_2_5_0_1():
    def func(k):
        return -1 / 24948 * k**10

    return func


def N_gg_2_5_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_5_0_2():
    def func(k):
        return -2 / 297297 * k**10

    return func


def N_gg_2_5_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_2_5_0_3():
    def func(k):
        return -16 / 57972915 * k**10

    return func


def N_gg_2_5_0_3(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * theta)
        + (819 / 256) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + 325 / 256
    )


def M_gg_2_5_2_0():
    def func(k):
        return -23 / 399168 * k**10

    return func


def N_gg_2_5_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_2_5_2_1():
    def func(k):
        return -23 / 399168 * k**10

    return func


def N_gg_2_5_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_2_5_2_2():
    def func(k):
        return -1 / 24948 * k**10

    return func


def N_gg_2_5_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_2_5_2_3():
    def func(k):
        return -59 / 3567564 * k**10

    return func


def N_gg_2_5_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_2_5_2_4():
    def func(k):
        return -59 / 3567564 * k**10

    return func


def N_gg_2_5_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_5_2_5():
    def func(k):
        return -2 / 297297 * k**10

    return func


def N_gg_2_5_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_2_5_2_6():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_2_5_2_7():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_2_5_2_8():
    def func(k):
        return -16 / 57972915 * k**10

    return func


def N_gg_2_5_2_8(theta, phi):
    return (
        -6825 / 5632 * np.cos(2 * theta)
        + (819 / 2816) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + (819 / 512) * np.cos(2 * phi - 5 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 512) * np.cos(2 * phi + 5 * theta)
        - 2275 / 2816
    )


def M_gg_2_5_2_9():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_2_9(theta, phi):
    return (
        (4725 / 2048) * np.cos(2 * theta)
        + (2079 / 1024) * np.cos(4 * theta)
        + (3003 / 2048) * np.cos(6 * theta)
        + (693 / 4096) * np.cos(2 * phi - 5 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (9009 / 2048) * np.cos(2 * phi + 5 * theta)
        + (45045 / 4096) * np.cos(2 * phi + 7 * theta)
        + 1225 / 1024
    )


def M_gg_2_5_2_10():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_2_10(theta, phi):
    return (
        (4725 / 2048) * np.cos(2 * theta)
        + (2079 / 1024) * np.cos(4 * theta)
        + (3003 / 2048) * np.cos(6 * theta)
        + (45045 / 4096) * np.cos(2 * phi - 7 * theta)
        + (9009 / 2048) * np.cos(2 * phi - 5 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (693 / 4096) * np.cos(2 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_2_5_4_0():
    def func(k):
        return -31 / 1297296 * k**10

    return func


def N_gg_2_5_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_2_5_4_1():
    def func(k):
        return -1 / 24948 * k**10

    return func


def N_gg_2_5_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_5_4_2():
    def func(k):
        return -59 / 3567564 * k**10

    return func


def N_gg_2_5_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_5_4_3():
    def func(k):
        return -16 / 4459455 * k**10

    return func


def N_gg_2_5_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_2_5_4_4():
    def func(k):
        return -31 / 1297296 * k**10

    return func


def N_gg_2_5_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_2_5_4_5():
    def func(k):
        return -59 / 3567564 * k**10

    return func


def N_gg_2_5_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_5_4_6():
    def func(k):
        return -2 / 297297 * k**10

    return func


def N_gg_2_5_4_6(theta, phi):
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


def M_gg_2_5_4_7():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_2_5_4_8():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_4_8(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (77175 / 73216) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (19845 / 6656) * np.cos(2 * phi + 3 * theta)
        + (6615 / 2048) * np.cos(2 * phi + 5 * theta)
        + (77175 / 1171456) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(4 * phi + 4 * theta)
        + (99225 / 8192) * np.cos(4 * phi + 6 * theta)
        + 694575 / 585728
    )


def M_gg_2_5_4_9():
    def func(k):
        return -16 / 4459455 * k**10

    return func


def N_gg_2_5_4_9(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_2_5_4_10():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_4_10(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_2_5_4_11():
    def func(k):
        return -16 / 57972915 * k**10

    return func


def N_gg_2_5_4_11(theta, phi):
    return (
        (1003275 / 382976) * np.cos(4 * phi)
        + (12285 / 69632) * np.cos(2 * theta)
        - 22113 / 11968 * np.cos(4 * theta)
        + (243243 / 69632) * np.cos(6 * theta)
        + (110565 / 34816) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(2 * phi + 5 * theta)
        + (85995 / 69632) * np.cos(4 * phi - 4 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (85995 / 69632) * np.cos(4 * phi + 4 * theta)
        + 61425 / 95744
    )


def M_gg_2_5_4_12():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_4_12(theta, phi):
    return (
        (694575 / 428032) * np.cos(4 * phi)
        - 552825 / 856064 * np.cos(2 * theta)
        + (49329 / 38912) * np.cos(4 * theta)
        + (243243 / 77824) * np.cos(6 * theta)
        + (93555 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        + (1289925 / 856064) * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        - 104895 / 77824 * np.cos(2 * phi + 3 * theta)
        - 36855 / 19456 * np.cos(2 * phi + 5 * theta)
        + (405405 / 77824) * np.cos(2 * phi + 7 * theta)
        + (19845 / 77824) * np.cos(4 * phi - 4 * theta)
        + (694575 / 856064) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 6 * theta)
        - 297675 / 428032
    )


def M_gg_2_5_4_13():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_4_13(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (6615 / 2048) * np.cos(2 * phi - 5 * theta)
        + (19845 / 6656) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (77175 / 73216) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (99225 / 8192) * np.cos(4 * phi - 6 * theta)
        + (6615 / 2048) * np.cos(4 * phi - 4 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (77175 / 1171456) * np.cos(4 * phi + 2 * theta)
        + 694575 / 585728
    )


def M_gg_2_5_4_14():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_4_14(theta, phi):
    return (
        (694575 / 428032) * np.cos(4 * phi)
        - 552825 / 856064 * np.cos(2 * theta)
        + (49329 / 38912) * np.cos(4 * theta)
        + (243243 / 77824) * np.cos(6 * theta)
        + (405405 / 77824) * np.cos(2 * phi - 7 * theta)
        - 36855 / 19456 * np.cos(2 * phi - 5 * theta)
        - 104895 / 77824 * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        + (1289925 / 856064) * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(2 * phi + 5 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 6 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 4 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (694575 / 856064) * np.cos(4 * phi + 2 * theta)
        + (19845 / 77824) * np.cos(4 * phi + 4 * theta)
        - 297675 / 428032
    )


def M_gg_2_5_6_0():
    def func(k):
        return -17 / 3243240 * k**10

    return func


def N_gg_2_5_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_2_5_6_1():
    def func(k):
        return -59 / 3567564 * k**10

    return func


def N_gg_2_5_6_1(theta, phi):
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


def M_gg_2_5_6_2():
    def func(k):
        return -16 / 4459455 * k**10

    return func


def N_gg_2_5_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_2_5_6_3():
    def func(k):
        return -2 / 6891885 * k**10

    return func


def N_gg_2_5_6_3(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (9009 / 2048) * np.cos(4 * phi + 4 * theta)
        + (693 / 4096) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(6 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_2_5_6_4():
    def func(k):
        return -59 / 3567564 * k**10

    return func


def N_gg_2_5_6_4(theta, phi):
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


def M_gg_2_5_6_5():
    def func(k):
        return -2 / 297297 * k**10

    return func


def N_gg_2_5_6_5(theta, phi):
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


def M_gg_2_5_6_6():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_6_6(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_2_5_6_7():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_6_7(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (1289925 / 856064) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        - 104895 / 77824 * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(2 * phi + 5 * theta)
        + (694575 / 856064) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        - 36855 / 19456 * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 6 * theta)
        + (19845 / 77824) * np.cos(6 * phi - theta)
        + (93555 / 77824) * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(6 * phi + 5 * theta)
        - 297675 / 428032
    )


def M_gg_2_5_6_8():
    def func(k):
        return -17 / 3243240 * k**10

    return func


def N_gg_2_5_6_8(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_2_5_6_9():
    def func(k):
        return -16 / 4459455 * k**10

    return func


def N_gg_2_5_6_9(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_2_5_6_10():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_6_10(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_2_5_6_11():
    def func(k):
        return -16 / 57972915 * k**10

    return func


def N_gg_2_5_6_11(theta, phi):
    return (
        -5589675 / 3638272 * np.cos(4 * phi)
        + (443625 / 661504) * np.cos(2 * theta)
        + (266175 / 909568) * np.cos(4 * theta)
        + (975975 / 661504) * np.cos(6 * theta)
        + (1863225 / 661504) * np.cos(2 * phi - 5 * theta)
        - 5589675 / 3638272 * np.cos(2 * phi - 3 * theta)
        + (443625 / 661504) * np.cos(2 * phi - theta)
        + (443625 / 661504) * np.cos(2 * phi + theta)
        - 5589675 / 3638272 * np.cos(2 * phi + 3 * theta)
        + (1863225 / 661504) * np.cos(2 * phi + 5 * theta)
        + (1863225 / 661504) * np.cos(4 * phi - 4 * theta)
        + (266175 / 909568) * np.cos(4 * phi - 2 * theta)
        + (266175 / 909568) * np.cos(4 * phi + 2 * theta)
        + (1863225 / 661504) * np.cos(4 * phi + 4 * theta)
        + (975975 / 661504) * np.cos(6 * phi - 3 * theta)
        + (1863225 / 661504) * np.cos(6 * phi - theta)
        + (1863225 / 661504) * np.cos(6 * phi + theta)
        + (975975 / 661504) * np.cos(6 * phi + 3 * theta)
        - 528125 / 909568
    )


def M_gg_2_5_6_12():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_6_12(theta, phi):
    return (
        (429975 / 856064) * np.cos(4 * phi)
        - 716625 / 3424256 * np.cos(2 * theta)
        - 429975 / 311296 * np.cos(4 * theta)
        + (975975 / 311296) * np.cos(6 * theta)
        + (1576575 / 622592) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 16960125 / 13697024 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 311296) * np.cos(2 * phi + 3 * theta)
        - 266175 / 311296 * np.cos(2 * phi + 5 * theta)
        + (2927925 / 1245184) * np.cos(2 * phi + 7 * theta)
        + (429975 / 311296) * np.cos(4 * phi - 4 * theta)
        + (12755925 / 6848512) * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        - 266175 / 311296 * np.cos(4 * phi + 4 * theta)
        + (2395575 / 622592) * np.cos(4 * phi + 6 * theta)
        + (525525 / 1245184) * np.cos(6 * phi - 3 * theta)
        + (429975 / 311296) * np.cos(6 * phi - theta)
        + (1576575 / 622592) * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(6 * phi + 5 * theta)
        + 1990625 / 3424256
    )


def M_gg_2_5_6_13():
    def func(k):
        return -2 / 6891885 * k**10

    return func


def N_gg_2_5_6_13(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (9009 / 2048) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (45045 / 4096) * np.cos(6 * phi - 5 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (693 / 4096) * np.cos(6 * phi - theta)
        + 1225 / 1024
    )


def M_gg_2_5_6_14():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_6_14(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (257985 / 77824) * np.cos(2 * phi - 5 * theta)
        - 104895 / 77824 * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (1289925 / 856064) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 6 * theta)
        - 36855 / 19456 * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (694575 / 856064) * np.cos(4 * phi + 2 * theta)
        + (405405 / 77824) * np.cos(6 * phi - 5 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        + (93555 / 77824) * np.cos(6 * phi - theta)
        + (19845 / 77824) * np.cos(6 * phi + theta)
        - 297675 / 428032
    )


def M_gg_2_5_6_15():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_6_15(theta, phi):
    return (
        (429975 / 856064) * np.cos(4 * phi)
        - 716625 / 3424256 * np.cos(2 * theta)
        - 429975 / 311296 * np.cos(4 * theta)
        + (975975 / 311296) * np.cos(6 * theta)
        + (2927925 / 1245184) * np.cos(2 * phi - 7 * theta)
        - 266175 / 311296 * np.cos(2 * phi - 5 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 16960125 / 13697024 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (1576575 / 622592) * np.cos(2 * phi + 5 * theta)
        + (2395575 / 622592) * np.cos(4 * phi - 6 * theta)
        - 266175 / 311296 * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        + (12755925 / 6848512) * np.cos(4 * phi + 2 * theta)
        + (429975 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2927925 / 1245184) * np.cos(6 * phi - 5 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        + (1576575 / 622592) * np.cos(6 * phi - theta)
        + (429975 / 311296) * np.cos(6 * phi + theta)
        + (525525 / 1245184) * np.cos(6 * phi + 3 * theta)
        + 1990625 / 3424256
    )


def M_gg_2_5_8_0():
    def func(k):
        return -1 / 2297295 * k**10

    return func


def N_gg_2_5_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi + theta)
        + (11781 / 4096) * np.cos(4 * phi + 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi + 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi + 4 * theta)
        + 20825 / 16384
    )


def M_gg_2_5_8_1():
    def func(k):
        return -16 / 4459455 * k**10

    return func


def N_gg_2_5_8_1(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_5_8_2():
    def func(k):
        return -2 / 6891885 * k**10

    return func


def N_gg_2_5_8_2(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (187425 / 77824) * np.cos(2 * phi - theta)
        - 26775 / 19456 * np.cos(2 * phi + theta)
        + (176715 / 77824) * np.cos(2 * phi + 3 * theta)
        - 11781 / 19456 * np.cos(4 * phi + 2 * theta)
        + (153153 / 77824) * np.cos(4 * phi + 4 * theta)
        + (153153 / 77824) * np.cos(6 * phi + theta)
        + (21879 / 19456) * np.cos(6 * phi + 3 * theta)
        + (109395 / 77824) * np.cos(6 * phi + 5 * theta)
        + (109395 / 77824) * np.cos(8 * phi + 2 * theta)
        + (546975 / 77824) * np.cos(8 * phi + 4 * theta)
        - 62475 / 77824
    )


def M_gg_2_5_8_3():
    def func(k):
        return -2 / 297297 * k**10

    return func


def N_gg_2_5_8_3(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_2_5_8_4():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_2_5_8_5():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_8_5(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (5060475 / 2023424) * np.cos(2 * phi - 3 * theta)
        - 5060475 / 4046848 * np.cos(2 * phi - theta)
        + (2457945 / 4046848) * np.cos(2 * phi + theta)
        - 530145 / 4046848 * np.cos(2 * phi + 3 * theta)
        + (530145 / 311296) * np.cos(2 * phi + 5 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi - 2 * theta)
        - 954261 / 1011712 * np.cos(4 * phi + 2 * theta)
        + (530145 / 311296) * np.cos(4 * phi + 4 * theta)
        + (530145 / 622592) * np.cos(4 * phi + 6 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        - 590733 / 311296 * np.cos(6 * phi + 3 * theta)
        + (984555 / 311296) * np.cos(6 * phi + 5 * theta)
        + (984555 / 311296) * np.cos(8 * phi + 2 * theta)
        + (2953665 / 622592) * np.cos(8 * phi + 4 * theta)
        + 5060475 / 8093696
    )


def M_gg_2_5_8_6():
    def func(k):
        return -16 / 4459455 * k**10

    return func


def N_gg_2_5_8_6(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_5_8_7():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_8_7(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_2_5_8_8():
    def func(k):
        return -16 / 57972915 * k**10

    return func


def N_gg_2_5_8_8(theta, phi):
    return (
        (429975 / 311296) * np.cos(4 * phi)
        + (2395575 / 622592) * np.cos(8 * phi)
        - 16960125 / 13697024 * np.cos(2 * theta)
        + (12755925 / 6848512) * np.cos(4 * theta)
        + (525525 / 1245184) * np.cos(6 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (429975 / 311296) * np.cos(2 * phi + 5 * theta)
        + (1576575 / 622592) * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        + (1576575 / 622592) * np.cos(4 * phi + 4 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        - 266175 / 311296 * np.cos(6 * phi - theta)
        - 266175 / 311296 * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi - 2 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi + 2 * theta)
        + 1990625 / 3424256
    )


def M_gg_2_5_8_9():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_8_9(theta, phi):
    return (
        -6185025 / 7159808 * np.cos(4 * phi)
        + (34459425 / 14319616) * np.cos(8 * phi)
        + (12182625 / 14319616) * np.cos(2 * theta)
        - 6185025 / 14319616 * np.cos(4 * theta)
        + (26801775 / 14319616) * np.cos(6 * theta)
        + (18555075 / 7159808) * np.cos(2 * phi - 5 * theta)
        - 5060475 / 3579904 * np.cos(2 * phi - 3 * theta)
        + (12182625 / 14319616) * np.cos(2 * phi - theta)
        + (133875 / 7159808) * np.cos(2 * phi + theta)
        - 6185025 / 7159808 * np.cos(2 * phi + 3 * theta)
        + (11486475 / 7159808) * np.cos(2 * phi + 5 * theta)
        + (11486475 / 14319616) * np.cos(2 * phi + 7 * theta)
        + (18555075 / 7159808) * np.cos(4 * phi - 4 * theta)
        - 6185025 / 14319616 * np.cos(4 * phi - 2 * theta)
        + (294525 / 223744) * np.cos(4 * phi + 2 * theta)
        - 3828825 / 3579904 * np.cos(4 * phi + 4 * theta)
        + (34459425 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (26801775 / 14319616) * np.cos(6 * phi - 3 * theta)
        + (11486475 / 7159808) * np.cos(6 * phi - theta)
        - 3828825 / 3579904 * np.cos(6 * phi + theta)
        - 7110675 / 7159808 * np.cos(6 * phi + 3 * theta)
        + (49774725 / 14319616) * np.cos(6 * phi + 5 * theta)
        + (11486475 / 14319616) * np.cos(8 * phi - 2 * theta)
        + (49774725 / 14319616) * np.cos(8 * phi + 2 * theta)
        + (35553375 / 14319616) * np.cos(8 * phi + 4 * theta)
        - 7809375 / 14319616
    )


def M_gg_2_5_8_10():
    def func(k):
        return -1 / 2297295 * k**10

    return func


def N_gg_2_5_8_10(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi - theta)
        + (11781 / 4096) * np.cos(4 * phi - 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi - 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi - 4 * theta)
        + 20825 / 16384
    )


def M_gg_2_5_8_11():
    def func(k):
        return -2 / 6891885 * k**10

    return func


def N_gg_2_5_8_11(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (176715 / 77824) * np.cos(2 * phi - 3 * theta)
        - 26775 / 19456 * np.cos(2 * phi - theta)
        + (187425 / 77824) * np.cos(2 * phi + theta)
        + (153153 / 77824) * np.cos(4 * phi - 4 * theta)
        - 11781 / 19456 * np.cos(4 * phi - 2 * theta)
        + (109395 / 77824) * np.cos(6 * phi - 5 * theta)
        + (21879 / 19456) * np.cos(6 * phi - 3 * theta)
        + (153153 / 77824) * np.cos(6 * phi - theta)
        + (546975 / 77824) * np.cos(8 * phi - 4 * theta)
        + (109395 / 77824) * np.cos(8 * phi - 2 * theta)
        - 62475 / 77824
    )


def M_gg_2_5_8_12():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_8_12(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (530145 / 311296) * np.cos(2 * phi - 5 * theta)
        - 530145 / 4046848 * np.cos(2 * phi - 3 * theta)
        + (2457945 / 4046848) * np.cos(2 * phi - theta)
        - 5060475 / 4046848 * np.cos(2 * phi + theta)
        + (5060475 / 2023424) * np.cos(2 * phi + 3 * theta)
        + (530145 / 622592) * np.cos(4 * phi - 6 * theta)
        + (530145 / 311296) * np.cos(4 * phi - 4 * theta)
        - 954261 / 1011712 * np.cos(4 * phi - 2 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi + 2 * theta)
        + (984555 / 311296) * np.cos(6 * phi - 5 * theta)
        - 590733 / 311296 * np.cos(6 * phi - 3 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        + (2953665 / 622592) * np.cos(8 * phi - 4 * theta)
        + (984555 / 311296) * np.cos(8 * phi - 2 * theta)
        + 5060475 / 8093696
    )


def M_gg_2_5_8_13():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_8_13(theta, phi):
    return (
        -6185025 / 7159808 * np.cos(4 * phi)
        + (34459425 / 14319616) * np.cos(8 * phi)
        + (12182625 / 14319616) * np.cos(2 * theta)
        - 6185025 / 14319616 * np.cos(4 * theta)
        + (26801775 / 14319616) * np.cos(6 * theta)
        + (11486475 / 14319616) * np.cos(2 * phi - 7 * theta)
        + (11486475 / 7159808) * np.cos(2 * phi - 5 * theta)
        - 6185025 / 7159808 * np.cos(2 * phi - 3 * theta)
        + (133875 / 7159808) * np.cos(2 * phi - theta)
        + (12182625 / 14319616) * np.cos(2 * phi + theta)
        - 5060475 / 3579904 * np.cos(2 * phi + 3 * theta)
        + (18555075 / 7159808) * np.cos(2 * phi + 5 * theta)
        + (34459425 / 14319616) * np.cos(4 * phi - 6 * theta)
        - 3828825 / 3579904 * np.cos(4 * phi - 4 * theta)
        + (294525 / 223744) * np.cos(4 * phi - 2 * theta)
        - 6185025 / 14319616 * np.cos(4 * phi + 2 * theta)
        + (18555075 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (49774725 / 14319616) * np.cos(6 * phi - 5 * theta)
        - 7110675 / 7159808 * np.cos(6 * phi - 3 * theta)
        - 3828825 / 3579904 * np.cos(6 * phi - theta)
        + (11486475 / 7159808) * np.cos(6 * phi + theta)
        + (26801775 / 14319616) * np.cos(6 * phi + 3 * theta)
        + (35553375 / 14319616) * np.cos(8 * phi - 4 * theta)
        + (49774725 / 14319616) * np.cos(8 * phi - 2 * theta)
        + (11486475 / 14319616) * np.cos(8 * phi + 2 * theta)
        - 7809375 / 14319616
    )


def M_gg_2_5_10_0():
    def func(k):
        return -2 / 6891885 * k**10

    return func


def N_gg_2_5_10_0(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (1819125 / 1245184) * np.cos(2 * phi - theta)
        + (363825 / 155648) * np.cos(2 * phi + theta)
        + (363825 / 622592) * np.cos(2 * phi + 3 * theta)
        + (675675 / 311296) * np.cos(4 * phi + 2 * theta)
        + (96525 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 622592) * np.cos(6 * phi + theta)
        + (289575 / 155648) * np.cos(6 * phi + 3 * theta)
        + (289575 / 2490368) * np.cos(6 * phi + 5 * theta)
        + (1640925 / 311296) * np.cos(8 * phi + 2 * theta)
        + (1640925 / 1245184) * np.cos(8 * phi + 4 * theta)
        + (1640925 / 131072) * np.cos(10 * phi + 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_2_5_10_1():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_2_5_10_2():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_10_2(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (22920975 / 14319616) * np.cos(2 * phi - 3 * theta)
        + (363825 / 894976) * np.cos(2 * phi - theta)
        - 3274425 / 3579904 * np.cos(2 * phi + theta)
        + (12879405 / 7159808) * np.cos(2 * phi + 3 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi + 5 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi - 2 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi + 2 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (42567525 / 14319616) * np.cos(6 * phi - theta)
        - 13378365 / 7159808 * np.cos(6 * phi + theta)
        + (7123545 / 3579904) * np.cos(6 * phi + 3 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi + 5 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi + 2 * theta)
        + (44304975 / 14319616) * np.cos(8 * phi + 4 * theta)
        + (2297295 / 753664) * np.cos(10 * phi + theta)
        + (4922775 / 753664) * np.cos(10 * phi + 3 * theta)
        - 9823275 / 14319616
    )


def M_gg_2_5_10_3():
    def func(k):
        return -82 / 57972915 * k**10

    return func


def N_gg_2_5_10_3(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_2_5_10_4():
    def func(k):
        return -16 / 57972915 * k**10

    return func


def N_gg_2_5_10_4(theta, phi):
    return (
        -184459275 / 121716736 * np.cos(4 * phi)
        - 36891855 / 14319616 * np.cos(8 * phi)
        + (568856925 / 486866944) * np.cos(2 * theta)
        + (269800713 / 243433472) * np.cos(4 * theta)
        + (35756721 / 486866944) * np.cos(6 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi - 5 * theta)
        + (854188335 / 486866944) * np.cos(2 * phi - 3 * theta)
        - 99324225 / 243433472 * np.cos(2 * phi - theta)
        - 99324225 / 243433472 * np.cos(2 * phi + theta)
        + (854188335 / 486866944) * np.cos(2 * phi + 3 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi + 5 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi + 2 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi + 4 * theta)
        + (553377825 / 243433472) * np.cos(6 * phi - 3 * theta)
        - 110675565 / 486866944 * np.cos(6 * phi - theta)
        - 110675565 / 486866944 * np.cos(6 * phi + theta)
        + (553377825 / 243433472) * np.cos(6 * phi + 3 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi - 2 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi + 2 * theta)
        + (7378371 / 1507328) * np.cos(10 * phi - theta)
        + (7378371 / 1507328) * np.cos(10 * phi + theta)
        - 81265275 / 121716736
    )


def M_gg_2_5_10_5():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_10_5(theta, phi):
    return (
        (945945 / 894976) * np.cos(4 * phi)
        + (9648639 / 28639232) * np.cos(8 * phi)
        - 138442689 / 114556928 * np.cos(2 * theta)
        + (112771197 / 71598080) * np.cos(4 * theta)
        + (393323931 / 572784640) * np.cos(6 * theta)
        + (332812557 / 229113856) * np.cos(2 * phi - 5 * theta)
        + (48592467 / 114556928) * np.cos(2 * phi - 3 * theta)
        - 89137125 / 229113856 * np.cos(2 * phi - theta)
        + (14771295 / 57278464) * np.cos(2 * phi + theta)
        - 79968735 / 229113856 * np.cos(2 * phi + 3 * theta)
        + (194675481 / 114556928) * np.cos(2 * phi + 5 * theta)
        + (43702659 / 229113856) * np.cos(2 * phi + 7 * theta)
        + (131107977 / 57278464) * np.cos(4 * phi - 4 * theta)
        - 59594535 / 57278464 * np.cos(4 * phi - 2 * theta)
        - 34999965 / 28639232 * np.cos(4 * phi + 2 * theta)
        + (86080995 / 57278464) * np.cos(4 * phi + 4 * theta)
        + (51648597 / 57278464) * np.cos(4 * phi + 6 * theta)
        + (655539885 / 229113856) * np.cos(6 * phi - 3 * theta)
        - 171972801 / 114556928 * np.cos(6 * phi - theta)
        + (218513295 / 229113856) * np.cos(6 * phi + theta)
        - 36891855 / 57278464 * np.cos(6 * phi + 3 * theta)
        + (258242985 / 114556928) * np.cos(6 * phi + 5 * theta)
        + (318405087 / 114556928) * np.cos(8 * phi - 2 * theta)
        - 209053845 / 114556928 * np.cos(8 * phi + 2 * theta)
        + (209053845 / 57278464) * np.cos(8 * phi + 4 * theta)
        + (106135029 / 60293120) * np.cos(10 * phi - theta)
        + (125432307 / 30146560) * np.cos(10 * phi + theta)
        + (41810769 / 12058624) * np.cos(10 * phi + 3 * theta)
        + 32089365 / 57278464
    )


def M_gg_2_5_10_6():
    def func(k):
        return -2 / 6891885 * k**10

    return func


def N_gg_2_5_10_6(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (363825 / 622592) * np.cos(2 * phi - 3 * theta)
        + (363825 / 155648) * np.cos(2 * phi - theta)
        + (1819125 / 1245184) * np.cos(2 * phi + theta)
        + (96525 / 311296) * np.cos(4 * phi - 4 * theta)
        + (675675 / 311296) * np.cos(4 * phi - 2 * theta)
        + (289575 / 2490368) * np.cos(6 * phi - 5 * theta)
        + (289575 / 155648) * np.cos(6 * phi - 3 * theta)
        + (2027025 / 622592) * np.cos(6 * phi - theta)
        + (1640925 / 1245184) * np.cos(8 * phi - 4 * theta)
        + (1640925 / 311296) * np.cos(8 * phi - 2 * theta)
        + (1640925 / 131072) * np.cos(10 * phi - 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_2_5_10_7():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_10_7(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi - 5 * theta)
        + (12879405 / 7159808) * np.cos(2 * phi - 3 * theta)
        - 3274425 / 3579904 * np.cos(2 * phi - theta)
        + (363825 / 894976) * np.cos(2 * phi + theta)
        + (22920975 / 14319616) * np.cos(2 * phi + 3 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi - 6 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi - 4 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi - 2 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi + 2 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi - 5 * theta)
        + (7123545 / 3579904) * np.cos(6 * phi - 3 * theta)
        - 13378365 / 7159808 * np.cos(6 * phi - theta)
        + (42567525 / 14319616) * np.cos(6 * phi + theta)
        + (44304975 / 14319616) * np.cos(8 * phi - 4 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi - 2 * theta)
        + (4922775 / 753664) * np.cos(10 * phi - 3 * theta)
        + (2297295 / 753664) * np.cos(10 * phi - theta)
        - 9823275 / 14319616
    )


def M_gg_2_5_10_8():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_10_8(theta, phi):
    return (
        (945945 / 894976) * np.cos(4 * phi)
        + (9648639 / 28639232) * np.cos(8 * phi)
        - 138442689 / 114556928 * np.cos(2 * theta)
        + (112771197 / 71598080) * np.cos(4 * theta)
        + (393323931 / 572784640) * np.cos(6 * theta)
        + (43702659 / 229113856) * np.cos(2 * phi - 7 * theta)
        + (194675481 / 114556928) * np.cos(2 * phi - 5 * theta)
        - 79968735 / 229113856 * np.cos(2 * phi - 3 * theta)
        + (14771295 / 57278464) * np.cos(2 * phi - theta)
        - 89137125 / 229113856 * np.cos(2 * phi + theta)
        + (48592467 / 114556928) * np.cos(2 * phi + 3 * theta)
        + (332812557 / 229113856) * np.cos(2 * phi + 5 * theta)
        + (51648597 / 57278464) * np.cos(4 * phi - 6 * theta)
        + (86080995 / 57278464) * np.cos(4 * phi - 4 * theta)
        - 34999965 / 28639232 * np.cos(4 * phi - 2 * theta)
        - 59594535 / 57278464 * np.cos(4 * phi + 2 * theta)
        + (131107977 / 57278464) * np.cos(4 * phi + 4 * theta)
        + (258242985 / 114556928) * np.cos(6 * phi - 5 * theta)
        - 36891855 / 57278464 * np.cos(6 * phi - 3 * theta)
        + (218513295 / 229113856) * np.cos(6 * phi - theta)
        - 171972801 / 114556928 * np.cos(6 * phi + theta)
        + (655539885 / 229113856) * np.cos(6 * phi + 3 * theta)
        + (209053845 / 57278464) * np.cos(8 * phi - 4 * theta)
        - 209053845 / 114556928 * np.cos(8 * phi - 2 * theta)
        + (318405087 / 114556928) * np.cos(8 * phi + 2 * theta)
        + (41810769 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (125432307 / 30146560) * np.cos(10 * phi - theta)
        + (106135029 / 60293120) * np.cos(10 * phi + theta)
        + 32089365 / 57278464
    )


def M_gg_2_5_12_0():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_12_0(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (42567525 / 114556928) * np.cos(2 * phi - 3 * theta)
        + (212837625 / 114556928) * np.cos(2 * phi - theta)
        + (127702575 / 57278464) * np.cos(2 * phi + theta)
        + (42567525 / 57278464) * np.cos(2 * phi + 3 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi + 5 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi - 2 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi + 2 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi + 4 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi + 6 * theta)
        + (172297125 / 114556928) * np.cos(6 * phi - theta)
        + (172297125 / 57278464) * np.cos(6 * phi + theta)
        + (73841625 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi + 5 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi + 2 * theta)
        + (14768325 / 24117248) * np.cos(8 * phi + 4 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi + theta)
        + (34459425 / 12058624) * np.cos(10 * phi + 3 * theta)
        + (34459425 / 2097152) * np.cos(12 * phi + 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_2_5_12_1():
    def func(k):
        return -16 / 57972915 * k**10

    return func


def N_gg_2_5_12_1(theta, phi):
    return (
        (10145260125 / 3894935552) * np.cos(4 * phi)
        + (11594583 / 3014656) * np.cos(8 * phi)
        + (9018009 / 524288) * np.cos(12 * phi)
        + (5150670525 / 3894935552) * np.cos(2 * theta)
        + (206026821 / 973733888) * np.cos(4 * theta)
        + (22891869 / 3894935552) * np.cos(6 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi - 5 * theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi - 3 * theta)
        + (2029052025 / 973733888) * np.cos(2 * phi - theta)
        + (2029052025 / 973733888) * np.cos(2 * phi + theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi + 3 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi + 5 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi - 4 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi - 2 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi + 2 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi + 4 * theta)
        + (32207175 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (289864575 / 114556928) * np.cos(6 * phi - theta)
        + (289864575 / 114556928) * np.cos(6 * phi + theta)
        + (32207175 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (27054027 / 6029312) * np.cos(10 * phi - theta)
        + (27054027 / 6029312) * np.cos(10 * phi + theta)
        + 572296725 / 486866944
    )


def M_gg_2_5_12_2():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_12_2(theta, phi):
    return (
        -307432125 / 229113856 * np.cos(4 * phi)
        - 1990989 / 1507328 * np.cos(8 * phi)
        + (2433431 / 524288) * np.cos(12 * phi)
        + (225450225 / 229113856) * np.cos(2 * theta)
        + (9018009 / 7159808) * np.cos(4 * theta)
        + (33066033 / 229113856) * np.cos(6 * theta)
        + (99198099 / 229113856) * np.cos(2 * phi - 5 * theta)
        + (12297285 / 7159808) * np.cos(2 * phi - 3 * theta)
        - 61486425 / 229113856 * np.cos(2 * phi - theta)
        - 20495475 / 28639232 * np.cos(2 * phi + theta)
        + (381215835 / 229113856) * np.cos(2 * phi + 3 * theta)
        + (18621603 / 28639232) * np.cos(2 * phi + 5 * theta)
        + (6441435 / 229113856) * np.cos(2 * phi + 7 * theta)
        + (225450225 / 229113856) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (20495475 / 28639232) * np.cos(4 * phi + 2 * theta)
        + (342567225 / 229113856) * np.cos(4 * phi + 4 * theta)
        + (43918875 / 229113856) * np.cos(4 * phi + 6 * theta)
        + (425850425 / 229113856) * np.cos(6 * phi - 3 * theta)
        + (16591575 / 28639232) * np.cos(6 * phi - theta)
        - 282056775 / 229113856 * np.cos(6 * phi + theta)
        + (57675475 / 28639232) * np.cos(6 * phi + 3 * theta)
        + (82957875 / 114556928) * np.cos(6 * phi + 5 * theta)
        + (36501465 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (9954945 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (11851125 / 6029312) * np.cos(8 * phi + 4 * theta)
        + (51102051 / 12058624) * np.cos(10 * phi - theta)
        - 1990989 / 753664 * np.cos(10 * phi + theta)
        + (49774725 / 12058624) * np.cos(10 * phi + 3 * theta)
        + (3318315 / 524288) * np.cos(12 * phi + 2 * theta)
        - 75150075 / 114556928
    )


def M_gg_2_5_12_3():
    def func(k):
        return -8 / 75810735 * k**10

    return func


def N_gg_2_5_12_3(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi - 5 * theta)
        + (42567525 / 57278464) * np.cos(2 * phi - 3 * theta)
        + (127702575 / 57278464) * np.cos(2 * phi - theta)
        + (212837625 / 114556928) * np.cos(2 * phi + theta)
        + (42567525 / 114556928) * np.cos(2 * phi + 3 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi - 6 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi - 4 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi + 2 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi - 5 * theta)
        + (73841625 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (172297125 / 57278464) * np.cos(6 * phi - theta)
        + (172297125 / 114556928) * np.cos(6 * phi + theta)
        + (14768325 / 24117248) * np.cos(8 * phi - 4 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi - 2 * theta)
        + (34459425 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi - theta)
        + (34459425 / 2097152) * np.cos(12 * phi - 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_2_5_12_4():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_12_4(theta, phi):
    return (
        -307432125 / 229113856 * np.cos(4 * phi)
        - 1990989 / 1507328 * np.cos(8 * phi)
        + (2433431 / 524288) * np.cos(12 * phi)
        + (225450225 / 229113856) * np.cos(2 * theta)
        + (9018009 / 7159808) * np.cos(4 * theta)
        + (33066033 / 229113856) * np.cos(6 * theta)
        + (6441435 / 229113856) * np.cos(2 * phi - 7 * theta)
        + (18621603 / 28639232) * np.cos(2 * phi - 5 * theta)
        + (381215835 / 229113856) * np.cos(2 * phi - 3 * theta)
        - 20495475 / 28639232 * np.cos(2 * phi - theta)
        - 61486425 / 229113856 * np.cos(2 * phi + theta)
        + (12297285 / 7159808) * np.cos(2 * phi + 3 * theta)
        + (99198099 / 229113856) * np.cos(2 * phi + 5 * theta)
        + (43918875 / 229113856) * np.cos(4 * phi - 6 * theta)
        + (342567225 / 229113856) * np.cos(4 * phi - 4 * theta)
        + (20495475 / 28639232) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 114556928) * np.cos(4 * phi + 2 * theta)
        + (225450225 / 229113856) * np.cos(4 * phi + 4 * theta)
        + (82957875 / 114556928) * np.cos(6 * phi - 5 * theta)
        + (57675475 / 28639232) * np.cos(6 * phi - 3 * theta)
        - 282056775 / 229113856 * np.cos(6 * phi - theta)
        + (16591575 / 28639232) * np.cos(6 * phi + theta)
        + (425850425 / 229113856) * np.cos(6 * phi + 3 * theta)
        + (11851125 / 6029312) * np.cos(8 * phi - 4 * theta)
        + (9954945 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (36501465 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (49774725 / 12058624) * np.cos(10 * phi - 3 * theta)
        - 1990989 / 753664 * np.cos(10 * phi - theta)
        + (51102051 / 12058624) * np.cos(10 * phi + theta)
        + (3318315 / 524288) * np.cos(12 * phi - 2 * theta)
        - 75150075 / 114556928
    )


def M_gg_2_5_14_0():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_14_0(theta, phi):
    return (
        (2299592295 / 916455424) * np.cos(4 * phi)
        + (153306153 / 48234496) * np.cos(8 * phi)
        + (12167155 / 2097152) * np.cos(12 * phi)
        + (1289575287 / 916455424) * np.cos(2 * theta)
        + (1289575287 / 4582277120) * np.cos(4 * theta)
        + (61408347 / 4582277120) * np.cos(6 * theta)
        + (99198099 / 1832910848) * np.cos(2 * phi - 5 * theta)
        + (297594297 / 458227712) * np.cos(2 * phi - 3 * theta)
        + (7439857425 / 3665821696) * np.cos(2 * phi - theta)
        + (495990495 / 229113856) * np.cos(2 * phi + theta)
        + (1487971485 / 1832910848) * np.cos(2 * phi + 3 * theta)
        + (42513471 / 458227712) * np.cos(2 * phi + 5 * theta)
        + (14171157 / 7331643392) * np.cos(2 * phi + 7 * theta)
        + (153306153 / 916455424) * np.cos(4 * phi - 4 * theta)
        + (2299592295 / 1832910848) * np.cos(4 * phi - 2 * theta)
        + (766530765 / 458227712) * np.cos(4 * phi + 2 * theta)
        + (328513185 / 916455424) * np.cos(4 * phi + 4 * theta)
        + (65702637 / 3665821696) * np.cos(4 * phi + 6 * theta)
        + (85170085 / 192937984) * np.cos(6 * phi - 3 * theta)
        + (51102051 / 24117248) * np.cos(6 * phi - theta)
        + (255510255 / 96468992) * np.cos(6 * phi + theta)
        + (12167155 / 12058624) * np.cos(6 * phi + 3 * theta)
        + (36501465 / 385875968) * np.cos(6 * phi + 5 * theta)
        + (51102051 / 48234496) * np.cos(8 * phi - 2 * theta)
        + (109504395 / 48234496) * np.cos(8 * phi + 2 * theta)
        + (36501465 / 96468992) * np.cos(8 * phi + 4 * theta)
        + (51102051 / 20971520) * np.cos(10 * phi - theta)
        + (21900879 / 5242880) * np.cos(10 * phi + theta)
        + (21900879 / 16777216) * np.cos(10 * phi + 3 * theta)
        + (36501465 / 8388608) * np.cos(12 * phi + 2 * theta)
        + (328513185 / 16777216) * np.cos(14 * phi + theta)
        + 2149292145 / 1832910848
    )


def M_gg_2_5_14_1():
    def func(k):
        return -16 / 985539555 * k**10

    return func


def N_gg_2_5_14_1(theta, phi):
    return (
        (2299592295 / 916455424) * np.cos(4 * phi)
        + (153306153 / 48234496) * np.cos(8 * phi)
        + (12167155 / 2097152) * np.cos(12 * phi)
        + (1289575287 / 916455424) * np.cos(2 * theta)
        + (1289575287 / 4582277120) * np.cos(4 * theta)
        + (61408347 / 4582277120) * np.cos(6 * theta)
        + (14171157 / 7331643392) * np.cos(2 * phi - 7 * theta)
        + (42513471 / 458227712) * np.cos(2 * phi - 5 * theta)
        + (1487971485 / 1832910848) * np.cos(2 * phi - 3 * theta)
        + (495990495 / 229113856) * np.cos(2 * phi - theta)
        + (7439857425 / 3665821696) * np.cos(2 * phi + theta)
        + (297594297 / 458227712) * np.cos(2 * phi + 3 * theta)
        + (99198099 / 1832910848) * np.cos(2 * phi + 5 * theta)
        + (65702637 / 3665821696) * np.cos(4 * phi - 6 * theta)
        + (328513185 / 916455424) * np.cos(4 * phi - 4 * theta)
        + (766530765 / 458227712) * np.cos(4 * phi - 2 * theta)
        + (2299592295 / 1832910848) * np.cos(4 * phi + 2 * theta)
        + (153306153 / 916455424) * np.cos(4 * phi + 4 * theta)
        + (36501465 / 385875968) * np.cos(6 * phi - 5 * theta)
        + (12167155 / 12058624) * np.cos(6 * phi - 3 * theta)
        + (255510255 / 96468992) * np.cos(6 * phi - theta)
        + (51102051 / 24117248) * np.cos(6 * phi + theta)
        + (85170085 / 192937984) * np.cos(6 * phi + 3 * theta)
        + (36501465 / 96468992) * np.cos(8 * phi - 4 * theta)
        + (109504395 / 48234496) * np.cos(8 * phi - 2 * theta)
        + (51102051 / 48234496) * np.cos(8 * phi + 2 * theta)
        + (21900879 / 16777216) * np.cos(10 * phi - 3 * theta)
        + (21900879 / 5242880) * np.cos(10 * phi - theta)
        + (51102051 / 20971520) * np.cos(10 * phi + theta)
        + (36501465 / 8388608) * np.cos(12 * phi - 2 * theta)
        + (328513185 / 16777216) * np.cos(14 * phi - theta)
        + 2149292145 / 1832910848
    )


def M_gg_2_6_0_0():
    def func(k):
        return (1 / 186624) * k**12

    return func


def N_gg_2_6_0_0(theta, phi):
    return 1


def M_gg_2_6_0_1():
    def func(k):
        return (1 / 352836) * k**12

    return func


def N_gg_2_6_0_1(theta, phi):
    return (15 / 4) * np.cos(2 * theta) + 5 / 4


def M_gg_2_6_0_2():
    def func(k):
        return (1 / 1656369) * k**12

    return func


def N_gg_2_6_0_2(theta, phi):
    return (45 / 16) * np.cos(2 * theta) + (315 / 64) * np.cos(4 * theta) + 81 / 64


def M_gg_2_6_0_3():
    def func(k):
        return (16 / 372683025) * k**12

    return func


def N_gg_2_6_0_3(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * theta)
        + (819 / 256) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + 325 / 256
    )


def M_gg_2_6_0_4():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_0_4(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * theta)
        + (11781 / 4096) * np.cos(4 * theta)
        + (7293 / 2048) * np.cos(6 * theta)
        + (109395 / 16384) * np.cos(8 * theta)
        + 20825 / 16384
    )


def M_gg_2_6_2_0():
    def func(k):
        return (1 / 256608) * k**12

    return func


def N_gg_2_6_2_0(theta, phi):
    return (15 / 4) * np.cos(2 * phi + theta) + 5 / 4


def M_gg_2_6_2_1():
    def func(k):
        return (1 / 256608) * k**12

    return func


def N_gg_2_6_2_1(theta, phi):
    return (15 / 4) * np.cos(2 * phi - theta) + 5 / 4


def M_gg_2_6_2_2():
    def func(k):
        return (1 / 352836) * k**12

    return func


def N_gg_2_6_2_2(theta, phi):
    return (
        (75 / 28) * np.cos(2 * theta)
        + (75 / 28) * np.cos(2 * phi - theta)
        + (75 / 28) * np.cos(2 * phi + theta)
        - 25 / 28
    )


def M_gg_2_6_2_3():
    def func(k):
        return (1 / 764478) * k**12

    return func


def N_gg_2_6_2_3(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (135 / 224) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + (225 / 32) * np.cos(2 * phi + 3 * theta)
        + 135 / 112
    )


def M_gg_2_6_2_4():
    def func(k):
        return (1 / 764478) * k**12

    return func


def N_gg_2_6_2_4(theta, phi):
    return (
        (225 / 112) * np.cos(2 * theta)
        + (225 / 32) * np.cos(2 * phi - 3 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (135 / 224) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_6_2_5():
    def func(k):
        return (1 / 1656369) * k**12

    return func


def N_gg_2_6_2_5(theta, phi):
    return (
        -225 / 308 * np.cos(2 * theta)
        + (1575 / 352) * np.cos(4 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        - 2025 / 2464
    )


def M_gg_2_6_2_6():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_2_6(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (1575 / 5632) * np.cos(2 * phi - 3 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (4725 / 2816) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 512) * np.cos(2 * phi + 5 * theta)
        + 3375 / 2816
    )


def M_gg_2_6_2_7():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_2_7(theta, phi):
    return (
        (1575 / 704) * np.cos(2 * theta)
        + (4725 / 2816) * np.cos(4 * theta)
        + (4725 / 512) * np.cos(2 * phi - 5 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2816) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (1575 / 5632) * np.cos(2 * phi + 3 * theta)
        + 3375 / 2816
    )


def M_gg_2_6_2_8():
    def func(k):
        return (16 / 372683025) * k**12

    return func


def N_gg_2_6_2_8(theta, phi):
    return (
        -6825 / 5632 * np.cos(2 * theta)
        + (819 / 2816) * np.cos(4 * theta)
        + (3003 / 512) * np.cos(6 * theta)
        + (819 / 512) * np.cos(2 * phi - 5 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 512) * np.cos(2 * phi + 5 * theta)
        - 2275 / 2816
    )


def M_gg_2_6_2_9():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_2_9(theta, phi):
    return (
        (4725 / 2048) * np.cos(2 * theta)
        + (2079 / 1024) * np.cos(4 * theta)
        + (3003 / 2048) * np.cos(6 * theta)
        + (693 / 4096) * np.cos(2 * phi - 5 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (9009 / 2048) * np.cos(2 * phi + 5 * theta)
        + (45045 / 4096) * np.cos(2 * phi + 7 * theta)
        + 1225 / 1024
    )


def M_gg_2_6_2_10():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_2_10(theta, phi):
    return (
        (4725 / 2048) * np.cos(2 * theta)
        + (2079 / 1024) * np.cos(4 * theta)
        + (3003 / 2048) * np.cos(6 * theta)
        + (45045 / 4096) * np.cos(2 * phi - 7 * theta)
        + (9009 / 2048) * np.cos(2 * phi - 5 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (693 / 4096) * np.cos(2 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_2_6_2_11():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_2_11(theta, phi):
    return (
        -26775 / 19456 * np.cos(2 * theta)
        - 11781 / 19456 * np.cos(4 * theta)
        + (21879 / 19456) * np.cos(6 * theta)
        + (546975 / 77824) * np.cos(8 * theta)
        + (109395 / 77824) * np.cos(2 * phi - 7 * theta)
        + (153153 / 77824) * np.cos(2 * phi - 5 * theta)
        + (176715 / 77824) * np.cos(2 * phi - 3 * theta)
        + (187425 / 77824) * np.cos(2 * phi - theta)
        + (187425 / 77824) * np.cos(2 * phi + theta)
        + (176715 / 77824) * np.cos(2 * phi + 3 * theta)
        + (153153 / 77824) * np.cos(2 * phi + 5 * theta)
        + (109395 / 77824) * np.cos(2 * phi + 7 * theta)
        - 62475 / 77824
    )


def M_gg_2_6_4_0():
    def func(k):
        return (1 / 555984) * k**12

    return func


def N_gg_2_6_4_0(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi + theta)
        + (315 / 64) * np.cos(4 * phi + 2 * theta)
        + 81 / 64
    )


def M_gg_2_6_4_1():
    def func(k):
        return (1 / 352836) * k**12

    return func


def N_gg_2_6_4_1(theta, phi):
    return (
        (225 / 32) * np.cos(4 * phi)
        + (135 / 224) * np.cos(2 * theta)
        + (225 / 112) * np.cos(2 * phi - theta)
        + (225 / 112) * np.cos(2 * phi + theta)
        + 135 / 112
    )


def M_gg_2_6_4_2():
    def func(k):
        return (1 / 764478) * k**12

    return func


def N_gg_2_6_4_2(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (6075 / 2464) * np.cos(2 * phi - theta)
        - 225 / 308 * np.cos(2 * phi + theta)
        + (675 / 352) * np.cos(2 * phi + 3 * theta)
        + (1575 / 352) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_6_4_3():
    def func(k):
        return (2 / 5733585) * k**12

    return func


def N_gg_2_6_4_3(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (1125 / 1408) * np.cos(2 * phi - theta)
        + (1575 / 704) * np.cos(2 * phi + theta)
        + (4725 / 1408) * np.cos(2 * phi + 3 * theta)
        + (4725 / 2816) * np.cos(4 * phi + 2 * theta)
        + (4725 / 512) * np.cos(4 * phi + 4 * theta)
        + 3375 / 2816
    )


def M_gg_2_6_4_4():
    def func(k):
        return (1 / 555984) * k**12

    return func


def N_gg_2_6_4_4(theta, phi):
    return (
        (45 / 16) * np.cos(2 * phi - theta)
        + (315 / 64) * np.cos(4 * phi - 2 * theta)
        + 81 / 64
    )


def M_gg_2_6_4_5():
    def func(k):
        return (1 / 764478) * k**12

    return func


def N_gg_2_6_4_5(theta, phi):
    return (
        (675 / 352) * np.cos(4 * phi)
        + (6075 / 2464) * np.cos(2 * theta)
        + (675 / 352) * np.cos(2 * phi - 3 * theta)
        - 225 / 308 * np.cos(2 * phi - theta)
        + (6075 / 2464) * np.cos(2 * phi + theta)
        + (1575 / 352) * np.cos(4 * phi - 2 * theta)
        - 2025 / 2464
    )


def M_gg_2_6_4_6():
    def func(k):
        return (1 / 1656369) * k**12

    return func


def N_gg_2_6_4_6(theta, phi):
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


def M_gg_2_6_4_7():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_4_7(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (4725 / 2816) * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi - theta)
        - 2835 / 1408 * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(2 * phi + 5 * theta)
        + (1575 / 2816) * np.cos(4 * phi - 2 * theta)
        + (8505 / 2816) * np.cos(4 * phi + 2 * theta)
        + (945 / 256) * np.cos(4 * phi + 4 * theta)
        - 2025 / 2816
    )


def M_gg_2_6_4_8():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_4_8(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (77175 / 73216) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (19845 / 6656) * np.cos(2 * phi + 3 * theta)
        + (6615 / 2048) * np.cos(2 * phi + 5 * theta)
        + (77175 / 1171456) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(4 * phi + 4 * theta)
        + (99225 / 8192) * np.cos(4 * phi + 6 * theta)
        + 694575 / 585728
    )


def M_gg_2_6_4_9():
    def func(k):
        return (2 / 5733585) * k**12

    return func


def N_gg_2_6_4_9(theta, phi):
    return (
        (1575 / 5632) * np.cos(4 * phi)
        + (4725 / 2816) * np.cos(2 * theta)
        + (4725 / 1408) * np.cos(2 * phi - 3 * theta)
        + (1575 / 704) * np.cos(2 * phi - theta)
        + (1125 / 1408) * np.cos(2 * phi + theta)
        + (4725 / 512) * np.cos(4 * phi - 4 * theta)
        + (4725 / 2816) * np.cos(4 * phi - 2 * theta)
        + 3375 / 2816
    )


def M_gg_2_6_4_10():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_4_10(theta, phi):
    return (
        (4725 / 2816) * np.cos(4 * phi)
        + (8505 / 2816) * np.cos(4 * theta)
        + (945 / 256) * np.cos(2 * phi - 5 * theta)
        - 2835 / 1408 * np.cos(2 * phi - 3 * theta)
        + (225 / 128) * np.cos(2 * phi + theta)
        + (4725 / 2816) * np.cos(2 * phi + 3 * theta)
        + (945 / 256) * np.cos(4 * phi - 4 * theta)
        + (8505 / 2816) * np.cos(4 * phi - 2 * theta)
        + (1575 / 2816) * np.cos(4 * phi + 2 * theta)
        - 2025 / 2816
    )


def M_gg_2_6_4_11():
    def func(k):
        return (16 / 372683025) * k**12

    return func


def N_gg_2_6_4_11(theta, phi):
    return (
        (1003275 / 382976) * np.cos(4 * phi)
        + (12285 / 69632) * np.cos(2 * theta)
        - 22113 / 11968 * np.cos(4 * theta)
        + (243243 / 69632) * np.cos(6 * theta)
        + (110565 / 34816) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(2 * phi + 5 * theta)
        + (85995 / 69632) * np.cos(4 * phi - 4 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (85995 / 69632) * np.cos(4 * phi + 4 * theta)
        + 61425 / 95744
    )


def M_gg_2_6_4_12():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_4_12(theta, phi):
    return (
        (694575 / 428032) * np.cos(4 * phi)
        - 552825 / 856064 * np.cos(2 * theta)
        + (49329 / 38912) * np.cos(4 * theta)
        + (243243 / 77824) * np.cos(6 * theta)
        + (93555 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        + (1289925 / 856064) * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        - 104895 / 77824 * np.cos(2 * phi + 3 * theta)
        - 36855 / 19456 * np.cos(2 * phi + 5 * theta)
        + (405405 / 77824) * np.cos(2 * phi + 7 * theta)
        + (19845 / 77824) * np.cos(4 * phi - 4 * theta)
        + (694575 / 856064) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 6 * theta)
        - 297675 / 428032
    )


def M_gg_2_6_4_13():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_4_13(theta, phi):
    return (
        (99225 / 292864) * np.cos(4 * phi)
        + (297675 / 146432) * np.cos(2 * theta)
        + (59535 / 53248) * np.cos(4 * theta)
        + (6615 / 2048) * np.cos(2 * phi - 5 * theta)
        + (19845 / 6656) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (77175 / 73216) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (99225 / 8192) * np.cos(4 * phi - 6 * theta)
        + (6615 / 2048) * np.cos(4 * phi - 4 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (77175 / 1171456) * np.cos(4 * phi + 2 * theta)
        + 694575 / 585728
    )


def M_gg_2_6_4_14():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_4_14(theta, phi):
    return (
        (694575 / 428032) * np.cos(4 * phi)
        - 552825 / 856064 * np.cos(2 * theta)
        + (49329 / 38912) * np.cos(4 * theta)
        + (243243 / 77824) * np.cos(6 * theta)
        + (405405 / 77824) * np.cos(2 * phi - 7 * theta)
        - 36855 / 19456 * np.cos(2 * phi - 5 * theta)
        - 104895 / 77824 * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        + (1289925 / 856064) * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(2 * phi + 5 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 6 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 4 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (694575 / 856064) * np.cos(4 * phi + 2 * theta)
        + (19845 / 77824) * np.cos(4 * phi + 4 * theta)
        - 297675 / 428032
    )


def M_gg_2_6_4_15():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_4_15(theta, phi):
    return (
        (5060475 / 2023424) * np.cos(4 * phi)
        + (2457945 / 4046848) * np.cos(2 * theta)
        - 954261 / 1011712 * np.cos(4 * theta)
        - 590733 / 311296 * np.cos(6 * theta)
        + (2953665 / 622592) * np.cos(8 * theta)
        + (984555 / 311296) * np.cos(2 * phi - 7 * theta)
        + (530145 / 311296) * np.cos(2 * phi - 5 * theta)
        - 530145 / 4046848 * np.cos(2 * phi - 3 * theta)
        - 5060475 / 4046848 * np.cos(2 * phi - theta)
        - 5060475 / 4046848 * np.cos(2 * phi + theta)
        - 530145 / 4046848 * np.cos(2 * phi + 3 * theta)
        + (530145 / 311296) * np.cos(2 * phi + 5 * theta)
        + (984555 / 311296) * np.cos(2 * phi + 7 * theta)
        + (530145 / 622592) * np.cos(4 * phi - 6 * theta)
        + (530145 / 311296) * np.cos(4 * phi - 4 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi - 2 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi + 2 * theta)
        + (530145 / 311296) * np.cos(4 * phi + 4 * theta)
        + (530145 / 622592) * np.cos(4 * phi + 6 * theta)
        + 5060475 / 8093696
    )


def M_gg_2_6_6_0():
    def func(k):
        return (1 / 2084940) * k**12

    return func


def N_gg_2_6_6_0(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi + theta)
        + (819 / 256) * np.cos(4 * phi + 2 * theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        + 325 / 256
    )


def M_gg_2_6_6_1():
    def func(k):
        return (1 / 764478) * k**12

    return func


def N_gg_2_6_6_1(theta, phi):
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


def M_gg_2_6_6_2():
    def func(k):
        return (2 / 5733585) * k**12

    return func


def N_gg_2_6_6_2(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (6825 / 2816) * np.cos(2 * phi - theta)
        - 6825 / 5632 * np.cos(2 * phi + theta)
        + (12285 / 5632) * np.cos(2 * phi + 3 * theta)
        + (819 / 2816) * np.cos(4 * phi + 2 * theta)
        + (819 / 512) * np.cos(4 * phi + 4 * theta)
        + (819 / 512) * np.cos(6 * phi + theta)
        + (3003 / 512) * np.cos(6 * phi + 3 * theta)
        - 2275 / 2816
    )


def M_gg_2_6_6_3():
    def func(k):
        return (4 / 97470945) * k**12

    return func


def N_gg_2_6_6_3(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (3675 / 4096) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (10395 / 4096) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (9009 / 2048) * np.cos(4 * phi + 4 * theta)
        + (693 / 4096) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(6 * phi + 5 * theta)
        + 1225 / 1024
    )


def M_gg_2_6_6_4():
    def func(k):
        return (1 / 764478) * k**12

    return func


def N_gg_2_6_6_4(theta, phi):
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


def M_gg_2_6_6_5():
    def func(k):
        return (1 / 1656369) * k**12

    return func


def N_gg_2_6_6_5(theta, phi):
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


def M_gg_2_6_6_6():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_6_6(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (1003275 / 382976) * np.cos(2 * phi - 3 * theta)
        - 20475 / 17408 * np.cos(2 * phi - theta)
        + (12285 / 69632) * np.cos(2 * phi + theta)
        + (282555 / 382976) * np.cos(2 * phi + 3 * theta)
        + (85995 / 69632) * np.cos(2 * phi + 5 * theta)
        + (429975 / 191488) * np.cos(4 * phi - 2 * theta)
        - 22113 / 11968 * np.cos(4 * phi + 2 * theta)
        + (110565 / 34816) * np.cos(4 * phi + 4 * theta)
        + (85995 / 69632) * np.cos(6 * phi - theta)
        + (110565 / 34816) * np.cos(6 * phi + theta)
        + (243243 / 69632) * np.cos(6 * phi + 3 * theta)
        + 61425 / 95744
    )


def M_gg_2_6_6_7():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_6_7(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (1289925 / 856064) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        - 104895 / 77824 * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(2 * phi + 5 * theta)
        + (694575 / 856064) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        - 36855 / 19456 * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(4 * phi + 6 * theta)
        + (19845 / 77824) * np.cos(6 * phi - theta)
        + (93555 / 77824) * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(6 * phi + 5 * theta)
        - 297675 / 428032
    )


def M_gg_2_6_6_8():
    def func(k):
        return (1 / 2084940) * k**12

    return func


def N_gg_2_6_6_8(theta, phi):
    return (
        (1365 / 512) * np.cos(2 * phi - theta)
        + (819 / 256) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + 325 / 256
    )


def M_gg_2_6_6_9():
    def func(k):
        return (2 / 5733585) * k**12

    return func


def N_gg_2_6_6_9(theta, phi):
    return (
        (12285 / 5632) * np.cos(4 * phi)
        + (6825 / 2816) * np.cos(2 * theta)
        + (12285 / 5632) * np.cos(2 * phi - 3 * theta)
        - 6825 / 5632 * np.cos(2 * phi - theta)
        + (6825 / 2816) * np.cos(2 * phi + theta)
        + (819 / 512) * np.cos(4 * phi - 4 * theta)
        + (819 / 2816) * np.cos(4 * phi - 2 * theta)
        + (3003 / 512) * np.cos(6 * phi - 3 * theta)
        + (819 / 512) * np.cos(6 * phi - theta)
        - 2275 / 2816
    )


def M_gg_2_6_6_10():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_6_10(theta, phi):
    return (
        (282555 / 382976) * np.cos(4 * phi)
        - 20475 / 17408 * np.cos(2 * theta)
        + (429975 / 191488) * np.cos(4 * theta)
        + (85995 / 69632) * np.cos(2 * phi - 5 * theta)
        + (282555 / 382976) * np.cos(2 * phi - 3 * theta)
        + (12285 / 69632) * np.cos(2 * phi - theta)
        - 20475 / 17408 * np.cos(2 * phi + theta)
        + (1003275 / 382976) * np.cos(2 * phi + 3 * theta)
        + (110565 / 34816) * np.cos(4 * phi - 4 * theta)
        - 22113 / 11968 * np.cos(4 * phi - 2 * theta)
        + (429975 / 191488) * np.cos(4 * phi + 2 * theta)
        + (243243 / 69632) * np.cos(6 * phi - 3 * theta)
        + (110565 / 34816) * np.cos(6 * phi - theta)
        + (85995 / 69632) * np.cos(6 * phi + theta)
        + 61425 / 95744
    )


def M_gg_2_6_6_11():
    def func(k):
        return (16 / 372683025) * k**12

    return func


def N_gg_2_6_6_11(theta, phi):
    return (
        -5589675 / 3638272 * np.cos(4 * phi)
        + (443625 / 661504) * np.cos(2 * theta)
        + (266175 / 909568) * np.cos(4 * theta)
        + (975975 / 661504) * np.cos(6 * theta)
        + (1863225 / 661504) * np.cos(2 * phi - 5 * theta)
        - 5589675 / 3638272 * np.cos(2 * phi - 3 * theta)
        + (443625 / 661504) * np.cos(2 * phi - theta)
        + (443625 / 661504) * np.cos(2 * phi + theta)
        - 5589675 / 3638272 * np.cos(2 * phi + 3 * theta)
        + (1863225 / 661504) * np.cos(2 * phi + 5 * theta)
        + (1863225 / 661504) * np.cos(4 * phi - 4 * theta)
        + (266175 / 909568) * np.cos(4 * phi - 2 * theta)
        + (266175 / 909568) * np.cos(4 * phi + 2 * theta)
        + (1863225 / 661504) * np.cos(4 * phi + 4 * theta)
        + (975975 / 661504) * np.cos(6 * phi - 3 * theta)
        + (1863225 / 661504) * np.cos(6 * phi - theta)
        + (1863225 / 661504) * np.cos(6 * phi + theta)
        + (975975 / 661504) * np.cos(6 * phi + 3 * theta)
        - 528125 / 909568
    )


def M_gg_2_6_6_12():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_6_12(theta, phi):
    return (
        (429975 / 856064) * np.cos(4 * phi)
        - 716625 / 3424256 * np.cos(2 * theta)
        - 429975 / 311296 * np.cos(4 * theta)
        + (975975 / 311296) * np.cos(6 * theta)
        + (1576575 / 622592) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 16960125 / 13697024 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 311296) * np.cos(2 * phi + 3 * theta)
        - 266175 / 311296 * np.cos(2 * phi + 5 * theta)
        + (2927925 / 1245184) * np.cos(2 * phi + 7 * theta)
        + (429975 / 311296) * np.cos(4 * phi - 4 * theta)
        + (12755925 / 6848512) * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        - 266175 / 311296 * np.cos(4 * phi + 4 * theta)
        + (2395575 / 622592) * np.cos(4 * phi + 6 * theta)
        + (525525 / 1245184) * np.cos(6 * phi - 3 * theta)
        + (429975 / 311296) * np.cos(6 * phi - theta)
        + (1576575 / 622592) * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(6 * phi + 5 * theta)
        + 1990625 / 3424256
    )


def M_gg_2_6_6_13():
    def func(k):
        return (4 / 97470945) * k**12

    return func


def N_gg_2_6_6_13(theta, phi):
    return (
        (945 / 2048) * np.cos(4 * phi)
        + (1575 / 1024) * np.cos(2 * theta)
        + (10395 / 4096) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (3675 / 4096) * np.cos(2 * phi + theta)
        + (9009 / 2048) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (45045 / 4096) * np.cos(6 * phi - 5 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (693 / 4096) * np.cos(6 * phi - theta)
        + 1225 / 1024
    )


def M_gg_2_6_6_14():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_6_14(theta, phi):
    return (
        (394065 / 214016) * np.cos(4 * phi)
        + (14175 / 53504) * np.cos(2 * theta)
        + (99225 / 38912) * np.cos(4 * theta)
        + (257985 / 77824) * np.cos(2 * phi - 5 * theta)
        - 104895 / 77824 * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (1289925 / 856064) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (257985 / 77824) * np.cos(4 * phi - 6 * theta)
        - 36855 / 19456 * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (694575 / 856064) * np.cos(4 * phi + 2 * theta)
        + (405405 / 77824) * np.cos(6 * phi - 5 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        + (93555 / 77824) * np.cos(6 * phi - theta)
        + (19845 / 77824) * np.cos(6 * phi + theta)
        - 297675 / 428032
    )


def M_gg_2_6_6_15():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_6_15(theta, phi):
    return (
        (429975 / 856064) * np.cos(4 * phi)
        - 716625 / 3424256 * np.cos(2 * theta)
        - 429975 / 311296 * np.cos(4 * theta)
        + (975975 / 311296) * np.cos(6 * theta)
        + (2927925 / 1245184) * np.cos(2 * phi - 7 * theta)
        - 266175 / 311296 * np.cos(2 * phi - 5 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 16960125 / 13697024 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (1576575 / 622592) * np.cos(2 * phi + 5 * theta)
        + (2395575 / 622592) * np.cos(4 * phi - 6 * theta)
        - 266175 / 311296 * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        + (12755925 / 6848512) * np.cos(4 * phi + 2 * theta)
        + (429975 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2927925 / 1245184) * np.cos(6 * phi - 5 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        + (1576575 / 622592) * np.cos(6 * phi - theta)
        + (429975 / 311296) * np.cos(6 * phi + theta)
        + (525525 / 1245184) * np.cos(6 * phi + 3 * theta)
        + 1990625 / 3424256
    )


def M_gg_2_6_6_16():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_6_16(theta, phi):
    return (
        -5060475 / 3579904 * np.cos(4 * phi)
        + (133875 / 7159808) * np.cos(2 * theta)
        + (294525 / 223744) * np.cos(4 * theta)
        - 7110675 / 7159808 * np.cos(6 * theta)
        + (35553375 / 14319616) * np.cos(8 * theta)
        + (49774725 / 14319616) * np.cos(2 * phi - 7 * theta)
        - 3828825 / 3579904 * np.cos(2 * phi - 5 * theta)
        - 6185025 / 7159808 * np.cos(2 * phi - 3 * theta)
        + (12182625 / 14319616) * np.cos(2 * phi - theta)
        + (12182625 / 14319616) * np.cos(2 * phi + theta)
        - 6185025 / 7159808 * np.cos(2 * phi + 3 * theta)
        - 3828825 / 3579904 * np.cos(2 * phi + 5 * theta)
        + (49774725 / 14319616) * np.cos(2 * phi + 7 * theta)
        + (34459425 / 14319616) * np.cos(4 * phi - 6 * theta)
        + (11486475 / 7159808) * np.cos(4 * phi - 4 * theta)
        - 6185025 / 14319616 * np.cos(4 * phi - 2 * theta)
        - 6185025 / 14319616 * np.cos(4 * phi + 2 * theta)
        + (11486475 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (34459425 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (11486475 / 14319616) * np.cos(6 * phi - 5 * theta)
        + (26801775 / 14319616) * np.cos(6 * phi - 3 * theta)
        + (18555075 / 7159808) * np.cos(6 * phi - theta)
        + (18555075 / 7159808) * np.cos(6 * phi + theta)
        + (26801775 / 14319616) * np.cos(6 * phi + 3 * theta)
        + (11486475 / 14319616) * np.cos(6 * phi + 5 * theta)
        - 7809375 / 14319616
    )


def M_gg_2_6_8_0():
    def func(k):
        return (1 / 17721990) * k**12

    return func


def N_gg_2_6_8_0(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi + theta)
        + (11781 / 4096) * np.cos(4 * phi + 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi + 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi + 4 * theta)
        + 20825 / 16384
    )


def M_gg_2_6_8_1():
    def func(k):
        return (2 / 5733585) * k**12

    return func


def N_gg_2_6_8_1(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (1575 / 1024) * np.cos(2 * phi - theta)
        + (4725 / 2048) * np.cos(2 * phi + theta)
        + (945 / 2048) * np.cos(2 * phi + 3 * theta)
        + (2079 / 1024) * np.cos(4 * phi + 2 * theta)
        + (693 / 4096) * np.cos(4 * phi + 4 * theta)
        + (9009 / 2048) * np.cos(6 * phi + theta)
        + (3003 / 2048) * np.cos(6 * phi + 3 * theta)
        + (45045 / 4096) * np.cos(8 * phi + 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_6_8_2():
    def func(k):
        return (4 / 97470945) * k**12

    return func


def N_gg_2_6_8_2(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (187425 / 77824) * np.cos(2 * phi - theta)
        - 26775 / 19456 * np.cos(2 * phi + theta)
        + (176715 / 77824) * np.cos(2 * phi + 3 * theta)
        - 11781 / 19456 * np.cos(4 * phi + 2 * theta)
        + (153153 / 77824) * np.cos(4 * phi + 4 * theta)
        + (153153 / 77824) * np.cos(6 * phi + theta)
        + (21879 / 19456) * np.cos(6 * phi + 3 * theta)
        + (109395 / 77824) * np.cos(6 * phi + 5 * theta)
        + (109395 / 77824) * np.cos(8 * phi + 2 * theta)
        + (546975 / 77824) * np.cos(8 * phi + 4 * theta)
        - 62475 / 77824
    )


def M_gg_2_6_8_3():
    def func(k):
        return (1 / 1656369) * k**12

    return func


def N_gg_2_6_8_3(theta, phi):
    return (
        (19845 / 6656) * np.cos(4 * phi)
        + (99225 / 8192) * np.cos(8 * phi)
        + (77175 / 73216) * np.cos(2 * theta)
        + (77175 / 1171456) * np.cos(4 * theta)
        + (99225 / 292864) * np.cos(2 * phi - 3 * theta)
        + (297675 / 146432) * np.cos(2 * phi - theta)
        + (297675 / 146432) * np.cos(2 * phi + theta)
        + (99225 / 292864) * np.cos(2 * phi + 3 * theta)
        + (59535 / 53248) * np.cos(4 * phi - 2 * theta)
        + (59535 / 53248) * np.cos(4 * phi + 2 * theta)
        + (6615 / 2048) * np.cos(6 * phi - theta)
        + (6615 / 2048) * np.cos(6 * phi + theta)
        + 694575 / 585728
    )


def M_gg_2_6_8_4():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_8_4(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (694575 / 428032) * np.cos(2 * phi - 3 * theta)
        + (14175 / 53504) * np.cos(2 * phi - theta)
        - 552825 / 856064 * np.cos(2 * phi + theta)
        + (394065 / 214016) * np.cos(2 * phi + 3 * theta)
        + (19845 / 77824) * np.cos(2 * phi + 5 * theta)
        + (99225 / 38912) * np.cos(4 * phi - 2 * theta)
        + (49329 / 38912) * np.cos(4 * phi + 2 * theta)
        + (93555 / 77824) * np.cos(4 * phi + 4 * theta)
        + (257985 / 77824) * np.cos(6 * phi - theta)
        - 36855 / 19456 * np.cos(6 * phi + theta)
        + (243243 / 77824) * np.cos(6 * phi + 3 * theta)
        + (405405 / 77824) * np.cos(8 * phi + 2 * theta)
        - 297675 / 428032
    )


def M_gg_2_6_8_5():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_8_5(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (5060475 / 2023424) * np.cos(2 * phi - 3 * theta)
        - 5060475 / 4046848 * np.cos(2 * phi - theta)
        + (2457945 / 4046848) * np.cos(2 * phi + theta)
        - 530145 / 4046848 * np.cos(2 * phi + 3 * theta)
        + (530145 / 311296) * np.cos(2 * phi + 5 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi - 2 * theta)
        - 954261 / 1011712 * np.cos(4 * phi + 2 * theta)
        + (530145 / 311296) * np.cos(4 * phi + 4 * theta)
        + (530145 / 622592) * np.cos(4 * phi + 6 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        - 590733 / 311296 * np.cos(6 * phi + 3 * theta)
        + (984555 / 311296) * np.cos(6 * phi + 5 * theta)
        + (984555 / 311296) * np.cos(8 * phi + 2 * theta)
        + (2953665 / 622592) * np.cos(8 * phi + 4 * theta)
        + 5060475 / 8093696
    )


def M_gg_2_6_8_6():
    def func(k):
        return (2 / 5733585) * k**12

    return func


def N_gg_2_6_8_6(theta, phi):
    return (
        (10395 / 4096) * np.cos(4 * phi)
        + (3675 / 4096) * np.cos(2 * theta)
        + (945 / 2048) * np.cos(2 * phi - 3 * theta)
        + (4725 / 2048) * np.cos(2 * phi - theta)
        + (1575 / 1024) * np.cos(2 * phi + theta)
        + (693 / 4096) * np.cos(4 * phi - 4 * theta)
        + (2079 / 1024) * np.cos(4 * phi - 2 * theta)
        + (3003 / 2048) * np.cos(6 * phi - 3 * theta)
        + (9009 / 2048) * np.cos(6 * phi - theta)
        + (45045 / 4096) * np.cos(8 * phi - 2 * theta)
        + 1225 / 1024
    )


def M_gg_2_6_8_7():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_8_7(theta, phi):
    return (
        -104895 / 77824 * np.cos(4 * phi)
        + (257985 / 77824) * np.cos(8 * phi)
        + (1289925 / 856064) * np.cos(2 * theta)
        + (694575 / 856064) * np.cos(4 * theta)
        + (19845 / 77824) * np.cos(2 * phi - 5 * theta)
        + (394065 / 214016) * np.cos(2 * phi - 3 * theta)
        - 552825 / 856064 * np.cos(2 * phi - theta)
        + (14175 / 53504) * np.cos(2 * phi + theta)
        + (694575 / 428032) * np.cos(2 * phi + 3 * theta)
        + (93555 / 77824) * np.cos(4 * phi - 4 * theta)
        + (49329 / 38912) * np.cos(4 * phi - 2 * theta)
        + (99225 / 38912) * np.cos(4 * phi + 2 * theta)
        + (243243 / 77824) * np.cos(6 * phi - 3 * theta)
        - 36855 / 19456 * np.cos(6 * phi - theta)
        + (257985 / 77824) * np.cos(6 * phi + theta)
        + (405405 / 77824) * np.cos(8 * phi - 2 * theta)
        - 297675 / 428032
    )


def M_gg_2_6_8_8():
    def func(k):
        return (16 / 372683025) * k**12

    return func


def N_gg_2_6_8_8(theta, phi):
    return (
        (429975 / 311296) * np.cos(4 * phi)
        + (2395575 / 622592) * np.cos(8 * phi)
        - 16960125 / 13697024 * np.cos(2 * theta)
        + (12755925 / 6848512) * np.cos(4 * theta)
        + (525525 / 1245184) * np.cos(6 * theta)
        + (429975 / 311296) * np.cos(2 * phi - 5 * theta)
        + (429975 / 856064) * np.cos(2 * phi - 3 * theta)
        - 716625 / 3424256 * np.cos(2 * phi - theta)
        - 716625 / 3424256 * np.cos(2 * phi + theta)
        + (429975 / 856064) * np.cos(2 * phi + 3 * theta)
        + (429975 / 311296) * np.cos(2 * phi + 5 * theta)
        + (1576575 / 622592) * np.cos(4 * phi - 4 * theta)
        - 429975 / 311296 * np.cos(4 * phi - 2 * theta)
        - 429975 / 311296 * np.cos(4 * phi + 2 * theta)
        + (1576575 / 622592) * np.cos(4 * phi + 4 * theta)
        + (975975 / 311296) * np.cos(6 * phi - 3 * theta)
        - 266175 / 311296 * np.cos(6 * phi - theta)
        - 266175 / 311296 * np.cos(6 * phi + theta)
        + (975975 / 311296) * np.cos(6 * phi + 3 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi - 2 * theta)
        + (2927925 / 1245184) * np.cos(8 * phi + 2 * theta)
        + 1990625 / 3424256
    )


def M_gg_2_6_8_9():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_8_9(theta, phi):
    return (
        -6185025 / 7159808 * np.cos(4 * phi)
        + (34459425 / 14319616) * np.cos(8 * phi)
        + (12182625 / 14319616) * np.cos(2 * theta)
        - 6185025 / 14319616 * np.cos(4 * theta)
        + (26801775 / 14319616) * np.cos(6 * theta)
        + (18555075 / 7159808) * np.cos(2 * phi - 5 * theta)
        - 5060475 / 3579904 * np.cos(2 * phi - 3 * theta)
        + (12182625 / 14319616) * np.cos(2 * phi - theta)
        + (133875 / 7159808) * np.cos(2 * phi + theta)
        - 6185025 / 7159808 * np.cos(2 * phi + 3 * theta)
        + (11486475 / 7159808) * np.cos(2 * phi + 5 * theta)
        + (11486475 / 14319616) * np.cos(2 * phi + 7 * theta)
        + (18555075 / 7159808) * np.cos(4 * phi - 4 * theta)
        - 6185025 / 14319616 * np.cos(4 * phi - 2 * theta)
        + (294525 / 223744) * np.cos(4 * phi + 2 * theta)
        - 3828825 / 3579904 * np.cos(4 * phi + 4 * theta)
        + (34459425 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (26801775 / 14319616) * np.cos(6 * phi - 3 * theta)
        + (11486475 / 7159808) * np.cos(6 * phi - theta)
        - 3828825 / 3579904 * np.cos(6 * phi + theta)
        - 7110675 / 7159808 * np.cos(6 * phi + 3 * theta)
        + (49774725 / 14319616) * np.cos(6 * phi + 5 * theta)
        + (11486475 / 14319616) * np.cos(8 * phi - 2 * theta)
        + (49774725 / 14319616) * np.cos(8 * phi + 2 * theta)
        + (35553375 / 14319616) * np.cos(8 * phi + 4 * theta)
        - 7809375 / 14319616
    )


def M_gg_2_6_8_10():
    def func(k):
        return (1 / 17721990) * k**12

    return func


def N_gg_2_6_8_10(theta, phi):
    return (
        (5355 / 2048) * np.cos(2 * phi - theta)
        + (11781 / 4096) * np.cos(4 * phi - 2 * theta)
        + (7293 / 2048) * np.cos(6 * phi - 3 * theta)
        + (109395 / 16384) * np.cos(8 * phi - 4 * theta)
        + 20825 / 16384
    )


def M_gg_2_6_8_11():
    def func(k):
        return (4 / 97470945) * k**12

    return func


def N_gg_2_6_8_11(theta, phi):
    return (
        (176715 / 77824) * np.cos(4 * phi)
        + (187425 / 77824) * np.cos(2 * theta)
        + (176715 / 77824) * np.cos(2 * phi - 3 * theta)
        - 26775 / 19456 * np.cos(2 * phi - theta)
        + (187425 / 77824) * np.cos(2 * phi + theta)
        + (153153 / 77824) * np.cos(4 * phi - 4 * theta)
        - 11781 / 19456 * np.cos(4 * phi - 2 * theta)
        + (109395 / 77824) * np.cos(6 * phi - 5 * theta)
        + (21879 / 19456) * np.cos(6 * phi - 3 * theta)
        + (153153 / 77824) * np.cos(6 * phi - theta)
        + (546975 / 77824) * np.cos(8 * phi - 4 * theta)
        + (109395 / 77824) * np.cos(8 * phi - 2 * theta)
        - 62475 / 77824
    )


def M_gg_2_6_8_12():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_8_12(theta, phi):
    return (
        -530145 / 4046848 * np.cos(4 * phi)
        + (530145 / 622592) * np.cos(8 * phi)
        - 5060475 / 4046848 * np.cos(2 * theta)
        + (18555075 / 8093696) * np.cos(4 * theta)
        + (530145 / 311296) * np.cos(2 * phi - 5 * theta)
        - 530145 / 4046848 * np.cos(2 * phi - 3 * theta)
        + (2457945 / 4046848) * np.cos(2 * phi - theta)
        - 5060475 / 4046848 * np.cos(2 * phi + theta)
        + (5060475 / 2023424) * np.cos(2 * phi + 3 * theta)
        + (530145 / 622592) * np.cos(4 * phi - 6 * theta)
        + (530145 / 311296) * np.cos(4 * phi - 4 * theta)
        - 954261 / 1011712 * np.cos(4 * phi - 2 * theta)
        + (18555075 / 8093696) * np.cos(4 * phi + 2 * theta)
        + (984555 / 311296) * np.cos(6 * phi - 5 * theta)
        - 590733 / 311296 * np.cos(6 * phi - 3 * theta)
        + (530145 / 311296) * np.cos(6 * phi - theta)
        + (530145 / 311296) * np.cos(6 * phi + theta)
        + (2953665 / 622592) * np.cos(8 * phi - 4 * theta)
        + (984555 / 311296) * np.cos(8 * phi - 2 * theta)
        + 5060475 / 8093696
    )


def M_gg_2_6_8_13():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_8_13(theta, phi):
    return (
        -6185025 / 7159808 * np.cos(4 * phi)
        + (34459425 / 14319616) * np.cos(8 * phi)
        + (12182625 / 14319616) * np.cos(2 * theta)
        - 6185025 / 14319616 * np.cos(4 * theta)
        + (26801775 / 14319616) * np.cos(6 * theta)
        + (11486475 / 14319616) * np.cos(2 * phi - 7 * theta)
        + (11486475 / 7159808) * np.cos(2 * phi - 5 * theta)
        - 6185025 / 7159808 * np.cos(2 * phi - 3 * theta)
        + (133875 / 7159808) * np.cos(2 * phi - theta)
        + (12182625 / 14319616) * np.cos(2 * phi + theta)
        - 5060475 / 3579904 * np.cos(2 * phi + 3 * theta)
        + (18555075 / 7159808) * np.cos(2 * phi + 5 * theta)
        + (34459425 / 14319616) * np.cos(4 * phi - 6 * theta)
        - 3828825 / 3579904 * np.cos(4 * phi - 4 * theta)
        + (294525 / 223744) * np.cos(4 * phi - 2 * theta)
        - 6185025 / 14319616 * np.cos(4 * phi + 2 * theta)
        + (18555075 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (49774725 / 14319616) * np.cos(6 * phi - 5 * theta)
        - 7110675 / 7159808 * np.cos(6 * phi - 3 * theta)
        - 3828825 / 3579904 * np.cos(6 * phi - theta)
        + (11486475 / 7159808) * np.cos(6 * phi + theta)
        + (26801775 / 14319616) * np.cos(6 * phi + 3 * theta)
        + (35553375 / 14319616) * np.cos(8 * phi - 4 * theta)
        + (49774725 / 14319616) * np.cos(8 * phi - 2 * theta)
        + (11486475 / 14319616) * np.cos(8 * phi + 2 * theta)
        - 7809375 / 14319616
    )


def M_gg_2_6_8_14():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_8_14(theta, phi):
    return (
        (441610785 / 372310016) * np.cos(4 * phi)
        + (693959805 / 229113856) * np.cos(8 * phi)
        - 825232275 / 1489240064 * np.cos(2 * theta)
        - 1226696625 / 2978480128 * np.cos(4 * theta)
        + (151876725 / 114556928) * np.cos(6 * theta)
        + (455630175 / 458227712) * np.cos(8 * theta)
        + (273378105 / 114556928) * np.cos(2 * phi - 7 * theta)
        - 63087255 / 57278464 * np.cos(2 * phi - 5 * theta)
        + (441610785 / 372310016) * np.cos(2 * phi - 3 * theta)
        - 825232275 / 1489240064 * np.cos(2 * phi - theta)
        - 825232275 / 1489240064 * np.cos(2 * phi + theta)
        + (441610785 / 372310016) * np.cos(2 * phi + 3 * theta)
        - 63087255 / 57278464 * np.cos(2 * phi + 5 * theta)
        + (273378105 / 114556928) * np.cos(2 * phi + 7 * theta)
        + (693959805 / 229113856) * np.cos(4 * phi - 6 * theta)
        - 63087255 / 57278464 * np.cos(4 * phi - 4 * theta)
        - 1226696625 / 2978480128 * np.cos(4 * phi - 2 * theta)
        - 1226696625 / 2978480128 * np.cos(4 * phi + 2 * theta)
        - 63087255 / 57278464 * np.cos(4 * phi + 4 * theta)
        + (693959805 / 229113856) * np.cos(4 * phi + 6 * theta)
        + (273378105 / 114556928) * np.cos(6 * phi - 5 * theta)
        + (151876725 / 114556928) * np.cos(6 * phi - 3 * theta)
        - 63087255 / 57278464 * np.cos(6 * phi - theta)
        - 63087255 / 57278464 * np.cos(6 * phi + theta)
        + (151876725 / 114556928) * np.cos(6 * phi + 3 * theta)
        + (273378105 / 114556928) * np.cos(6 * phi + 5 * theta)
        + (455630175 / 458227712) * np.cos(8 * phi - 4 * theta)
        + (273378105 / 114556928) * np.cos(8 * phi - 2 * theta)
        + (273378105 / 114556928) * np.cos(8 * phi + 2 * theta)
        + (455630175 / 458227712) * np.cos(8 * phi + 4 * theta)
        + 3035764375 / 5956960256
    )


def M_gg_2_6_10_0():
    def func(k):
        return (4 / 97470945) * k**12

    return func


def N_gg_2_6_10_0(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (1819125 / 1245184) * np.cos(2 * phi - theta)
        + (363825 / 155648) * np.cos(2 * phi + theta)
        + (363825 / 622592) * np.cos(2 * phi + 3 * theta)
        + (675675 / 311296) * np.cos(4 * phi + 2 * theta)
        + (96525 / 311296) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 622592) * np.cos(6 * phi + theta)
        + (289575 / 155648) * np.cos(6 * phi + 3 * theta)
        + (289575 / 2490368) * np.cos(6 * phi + 5 * theta)
        + (1640925 / 311296) * np.cos(8 * phi + 2 * theta)
        + (1640925 / 1245184) * np.cos(8 * phi + 4 * theta)
        + (1640925 / 131072) * np.cos(10 * phi + 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_2_6_10_1():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_10_1(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (7640325 / 21168128) * np.cos(2 * phi - 3 * theta)
        + (2546775 / 1323008) * np.cos(2 * phi - theta)
        + (22920975 / 10584064) * np.cos(2 * phi + theta)
        + (1528065 / 2646016) * np.cos(2 * phi + 3 * theta)
        + (509355 / 21168128) * np.cos(2 * phi + 5 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (945945 / 5292032) * np.cos(4 * phi + 4 * theta)
        + (42567525 / 21168128) * np.cos(6 * phi - theta)
        + (8513505 / 2646016) * np.cos(6 * phi + theta)
        + (8513505 / 10584064) * np.cos(6 * phi + 3 * theta)
        + (945945 / 311296) * np.cos(8 * phi + 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi + theta)
        + 6251175 / 5292032
    )


def M_gg_2_6_10_2():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_10_2(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (22920975 / 14319616) * np.cos(2 * phi - 3 * theta)
        + (363825 / 894976) * np.cos(2 * phi - theta)
        - 3274425 / 3579904 * np.cos(2 * phi + theta)
        + (12879405 / 7159808) * np.cos(2 * phi + 3 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi + 5 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi - 2 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi + 2 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi + 4 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi + 6 * theta)
        + (42567525 / 14319616) * np.cos(6 * phi - theta)
        - 13378365 / 7159808 * np.cos(6 * phi + theta)
        + (7123545 / 3579904) * np.cos(6 * phi + 3 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi + 5 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi + 2 * theta)
        + (44304975 / 14319616) * np.cos(8 * phi + 4 * theta)
        + (2297295 / 753664) * np.cos(10 * phi + theta)
        + (4922775 / 753664) * np.cos(10 * phi + 3 * theta)
        - 9823275 / 14319616
    )


def M_gg_2_6_10_3():
    def func(k):
        return (4 / 24845535) * k**12

    return func


def N_gg_2_6_10_3(theta, phi):
    return (
        (14189175 / 5292032) * np.cos(4 * phi)
        + (2837835 / 622592) * np.cos(8 * phi)
        + (6251175 / 5292032) * np.cos(2 * theta)
        + (1250235 / 10584064) * np.cos(4 * theta)
        + (509355 / 21168128) * np.cos(2 * phi - 5 * theta)
        + (1528065 / 2646016) * np.cos(2 * phi - 3 * theta)
        + (22920975 / 10584064) * np.cos(2 * phi - theta)
        + (2546775 / 1323008) * np.cos(2 * phi + theta)
        + (7640325 / 21168128) * np.cos(2 * phi + 3 * theta)
        + (945945 / 5292032) * np.cos(4 * phi - 4 * theta)
        + (8513505 / 5292032) * np.cos(4 * phi - 2 * theta)
        + (4729725 / 5292032) * np.cos(4 * phi + 2 * theta)
        + (8513505 / 10584064) * np.cos(6 * phi - 3 * theta)
        + (8513505 / 2646016) * np.cos(6 * phi - theta)
        + (42567525 / 21168128) * np.cos(6 * phi + theta)
        + (945945 / 311296) * np.cos(8 * phi - 2 * theta)
        + (945945 / 65536) * np.cos(10 * phi - theta)
        + 6251175 / 5292032
    )


def M_gg_2_6_10_4():
    def func(k):
        return (16 / 372683025) * k**12

    return func


def N_gg_2_6_10_4(theta, phi):
    return (
        -184459275 / 121716736 * np.cos(4 * phi)
        - 36891855 / 14319616 * np.cos(8 * phi)
        + (568856925 / 486866944) * np.cos(2 * theta)
        + (269800713 / 243433472) * np.cos(4 * theta)
        + (35756721 / 486866944) * np.cos(6 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi - 5 * theta)
        + (854188335 / 486866944) * np.cos(2 * phi - 3 * theta)
        - 99324225 / 243433472 * np.cos(2 * phi - theta)
        - 99324225 / 243433472 * np.cos(2 * phi + theta)
        + (854188335 / 486866944) * np.cos(2 * phi + 3 * theta)
        + (178783605 / 486866944) * np.cos(2 * phi + 5 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 121716736) * np.cos(4 * phi + 2 * theta)
        + (258242985 / 243433472) * np.cos(4 * phi + 4 * theta)
        + (553377825 / 243433472) * np.cos(6 * phi - 3 * theta)
        - 110675565 / 486866944 * np.cos(6 * phi - theta)
        - 110675565 / 486866944 * np.cos(6 * phi + theta)
        + (553377825 / 243433472) * np.cos(6 * phi + 3 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi - 2 * theta)
        + (110675565 / 28639232) * np.cos(8 * phi + 2 * theta)
        + (7378371 / 1507328) * np.cos(10 * phi - theta)
        + (7378371 / 1507328) * np.cos(10 * phi + theta)
        - 81265275 / 121716736
    )


def M_gg_2_6_10_5():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_10_5(theta, phi):
    return (
        (945945 / 894976) * np.cos(4 * phi)
        + (9648639 / 28639232) * np.cos(8 * phi)
        - 138442689 / 114556928 * np.cos(2 * theta)
        + (112771197 / 71598080) * np.cos(4 * theta)
        + (393323931 / 572784640) * np.cos(6 * theta)
        + (332812557 / 229113856) * np.cos(2 * phi - 5 * theta)
        + (48592467 / 114556928) * np.cos(2 * phi - 3 * theta)
        - 89137125 / 229113856 * np.cos(2 * phi - theta)
        + (14771295 / 57278464) * np.cos(2 * phi + theta)
        - 79968735 / 229113856 * np.cos(2 * phi + 3 * theta)
        + (194675481 / 114556928) * np.cos(2 * phi + 5 * theta)
        + (43702659 / 229113856) * np.cos(2 * phi + 7 * theta)
        + (131107977 / 57278464) * np.cos(4 * phi - 4 * theta)
        - 59594535 / 57278464 * np.cos(4 * phi - 2 * theta)
        - 34999965 / 28639232 * np.cos(4 * phi + 2 * theta)
        + (86080995 / 57278464) * np.cos(4 * phi + 4 * theta)
        + (51648597 / 57278464) * np.cos(4 * phi + 6 * theta)
        + (655539885 / 229113856) * np.cos(6 * phi - 3 * theta)
        - 171972801 / 114556928 * np.cos(6 * phi - theta)
        + (218513295 / 229113856) * np.cos(6 * phi + theta)
        - 36891855 / 57278464 * np.cos(6 * phi + 3 * theta)
        + (258242985 / 114556928) * np.cos(6 * phi + 5 * theta)
        + (318405087 / 114556928) * np.cos(8 * phi - 2 * theta)
        - 209053845 / 114556928 * np.cos(8 * phi + 2 * theta)
        + (209053845 / 57278464) * np.cos(8 * phi + 4 * theta)
        + (106135029 / 60293120) * np.cos(10 * phi - theta)
        + (125432307 / 30146560) * np.cos(10 * phi + theta)
        + (41810769 / 12058624) * np.cos(10 * phi + 3 * theta)
        + 32089365 / 57278464
    )


def M_gg_2_6_10_6():
    def func(k):
        return (4 / 97470945) * k**12

    return func


def N_gg_2_6_10_6(theta, phi):
    return (
        (675675 / 311296) * np.cos(4 * phi)
        + (297675 / 311296) * np.cos(2 * theta)
        + (363825 / 622592) * np.cos(2 * phi - 3 * theta)
        + (363825 / 155648) * np.cos(2 * phi - theta)
        + (1819125 / 1245184) * np.cos(2 * phi + theta)
        + (96525 / 311296) * np.cos(4 * phi - 4 * theta)
        + (675675 / 311296) * np.cos(4 * phi - 2 * theta)
        + (289575 / 2490368) * np.cos(6 * phi - 5 * theta)
        + (289575 / 155648) * np.cos(6 * phi - 3 * theta)
        + (2027025 / 622592) * np.cos(6 * phi - theta)
        + (1640925 / 1245184) * np.cos(8 * phi - 4 * theta)
        + (1640925 / 311296) * np.cos(8 * phi - 2 * theta)
        + (1640925 / 131072) * np.cos(10 * phi - 3 * theta)
        + 1488375 / 1245184
    )


def M_gg_2_6_10_7():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_10_7(theta, phi):
    return (
        -6081075 / 7159808 * np.cos(4 * phi)
        + (48243195 / 14319616) * np.cos(8 * phi)
        + (9823275 / 7159808) * np.cos(2 * theta)
        + (13752585 / 14319616) * np.cos(4 * theta)
        + (6621615 / 14319616) * np.cos(2 * phi - 5 * theta)
        + (12879405 / 7159808) * np.cos(2 * phi - 3 * theta)
        - 3274425 / 3579904 * np.cos(2 * phi - theta)
        + (363825 / 894976) * np.cos(2 * phi + theta)
        + (22920975 / 14319616) * np.cos(2 * phi + 3 * theta)
        + (2027025 / 14319616) * np.cos(4 * phi - 6 * theta)
        + (11563695 / 7159808) * np.cos(4 * phi - 4 * theta)
        + (1216215 / 3579904) * np.cos(4 * phi - 2 * theta)
        + (33108075 / 14319616) * np.cos(4 * phi + 2 * theta)
        + (13030875 / 14319616) * np.cos(6 * phi - 5 * theta)
        + (7123545 / 3579904) * np.cos(6 * phi - 3 * theta)
        - 13378365 / 7159808 * np.cos(6 * phi - theta)
        + (42567525 / 14319616) * np.cos(6 * phi + theta)
        + (44304975 / 14319616) * np.cos(8 * phi - 4 * theta)
        - 10173735 / 7159808 * np.cos(8 * phi - 2 * theta)
        + (4922775 / 753664) * np.cos(10 * phi - 3 * theta)
        + (2297295 / 753664) * np.cos(10 * phi - theta)
        - 9823275 / 14319616
    )


def M_gg_2_6_10_8():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_10_8(theta, phi):
    return (
        (945945 / 894976) * np.cos(4 * phi)
        + (9648639 / 28639232) * np.cos(8 * phi)
        - 138442689 / 114556928 * np.cos(2 * theta)
        + (112771197 / 71598080) * np.cos(4 * theta)
        + (393323931 / 572784640) * np.cos(6 * theta)
        + (43702659 / 229113856) * np.cos(2 * phi - 7 * theta)
        + (194675481 / 114556928) * np.cos(2 * phi - 5 * theta)
        - 79968735 / 229113856 * np.cos(2 * phi - 3 * theta)
        + (14771295 / 57278464) * np.cos(2 * phi - theta)
        - 89137125 / 229113856 * np.cos(2 * phi + theta)
        + (48592467 / 114556928) * np.cos(2 * phi + 3 * theta)
        + (332812557 / 229113856) * np.cos(2 * phi + 5 * theta)
        + (51648597 / 57278464) * np.cos(4 * phi - 6 * theta)
        + (86080995 / 57278464) * np.cos(4 * phi - 4 * theta)
        - 34999965 / 28639232 * np.cos(4 * phi - 2 * theta)
        - 59594535 / 57278464 * np.cos(4 * phi + 2 * theta)
        + (131107977 / 57278464) * np.cos(4 * phi + 4 * theta)
        + (258242985 / 114556928) * np.cos(6 * phi - 5 * theta)
        - 36891855 / 57278464 * np.cos(6 * phi - 3 * theta)
        + (218513295 / 229113856) * np.cos(6 * phi - theta)
        - 171972801 / 114556928 * np.cos(6 * phi + theta)
        + (655539885 / 229113856) * np.cos(6 * phi + 3 * theta)
        + (209053845 / 57278464) * np.cos(8 * phi - 4 * theta)
        - 209053845 / 114556928 * np.cos(8 * phi - 2 * theta)
        + (318405087 / 114556928) * np.cos(8 * phi + 2 * theta)
        + (41810769 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (125432307 / 30146560) * np.cos(10 * phi - theta)
        + (106135029 / 60293120) * np.cos(10 * phi + theta)
        + 32089365 / 57278464
    )


def M_gg_2_6_10_9():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_10_9(theta, phi):
    return (
        -16081065 / 14319616 * np.cos(4 * phi)
        - 200477277 / 114556928 * np.cos(8 * phi)
        + (13697019 / 14319616) * np.cos(2 * theta)
        - 555910047 / 572784640 * np.cos(4 * theta)
        + (67540473 / 35799040) * np.cos(6 * theta)
        + (67540473 / 229113856) * np.cos(8 * theta)
        + (247648401 / 229113856) * np.cos(2 * phi - 7 * theta)
        + (247648401 / 229113856) * np.cos(2 * phi - 5 * theta)
        - 167985279 / 229113856 * np.cos(2 * phi - 3 * theta)
        + (60613245 / 229113856) * np.cos(2 * phi - theta)
        + (60613245 / 229113856) * np.cos(2 * phi + theta)
        - 167985279 / 229113856 * np.cos(2 * phi + 3 * theta)
        + (247648401 / 229113856) * np.cos(2 * phi + 5 * theta)
        + (247648401 / 229113856) * np.cos(2 * phi + 7 * theta)
        + (247648401 / 114556928) * np.cos(4 * phi - 6 * theta)
        - 22513491 / 28639232 * np.cos(4 * phi - 4 * theta)
        + (112567455 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (112567455 / 114556928) * np.cos(4 * phi + 2 * theta)
        - 22513491 / 28639232 * np.cos(4 * phi + 4 * theta)
        + (247648401 / 114556928) * np.cos(4 * phi + 6 * theta)
        + (337702365 / 114556928) * np.cos(6 * phi - 5 * theta)
        - 337702365 / 229113856 * np.cos(6 * phi - 3 * theta)
        + (125432307 / 229113856) * np.cos(6 * phi - theta)
        + (125432307 / 229113856) * np.cos(6 * phi + theta)
        - 337702365 / 229113856 * np.cos(6 * phi + 3 * theta)
        + (337702365 / 114556928) * np.cos(6 * phi + 5 * theta)
        + (637882245 / 229113856) * np.cos(8 * phi - 4 * theta)
        + (18225207 / 28639232) * np.cos(8 * phi - 2 * theta)
        + (18225207 / 28639232) * np.cos(8 * phi + 2 * theta)
        + (637882245 / 229113856) * np.cos(8 * phi + 4 * theta)
        + (18225207 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (200477277 / 60293120) * np.cos(10 * phi - theta)
        + (200477277 / 60293120) * np.cos(10 * phi + theta)
        + (18225207 / 12058624) * np.cos(10 * phi + 3 * theta)
        - 115716195 / 229113856
    )


def M_gg_2_6_12_0():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_12_0(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (42567525 / 114556928) * np.cos(2 * phi - 3 * theta)
        + (212837625 / 114556928) * np.cos(2 * phi - theta)
        + (127702575 / 57278464) * np.cos(2 * phi + theta)
        + (42567525 / 57278464) * np.cos(2 * phi + 3 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi + 5 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi - 2 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi + 2 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi + 4 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi + 6 * theta)
        + (172297125 / 114556928) * np.cos(6 * phi - theta)
        + (172297125 / 57278464) * np.cos(6 * phi + theta)
        + (73841625 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi + 5 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi + 2 * theta)
        + (14768325 / 24117248) * np.cos(8 * phi + 4 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi + theta)
        + (34459425 / 12058624) * np.cos(10 * phi + 3 * theta)
        + (34459425 / 2097152) * np.cos(12 * phi + 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_2_6_12_1():
    def func(k):
        return (16 / 372683025) * k**12

    return func


def N_gg_2_6_12_1(theta, phi):
    return (
        (10145260125 / 3894935552) * np.cos(4 * phi)
        + (11594583 / 3014656) * np.cos(8 * phi)
        + (9018009 / 524288) * np.cos(12 * phi)
        + (5150670525 / 3894935552) * np.cos(2 * theta)
        + (206026821 / 973733888) * np.cos(4 * theta)
        + (22891869 / 3894935552) * np.cos(6 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi - 5 * theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi - 3 * theta)
        + (2029052025 / 973733888) * np.cos(2 * phi - theta)
        + (2029052025 / 973733888) * np.cos(2 * phi + theta)
        + (1217431215 / 1947467776) * np.cos(2 * phi + 3 * theta)
        + (81162081 / 1947467776) * np.cos(2 * phi + 5 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi - 4 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi - 2 * theta)
        + (676350675 / 486866944) * np.cos(4 * phi + 2 * theta)
        + (676350675 / 3894935552) * np.cos(4 * phi + 4 * theta)
        + (32207175 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (289864575 / 114556928) * np.cos(6 * phi - theta)
        + (289864575 / 114556928) * np.cos(6 * phi + theta)
        + (32207175 / 57278464) * np.cos(6 * phi + 3 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (19324305 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (27054027 / 6029312) * np.cos(10 * phi - theta)
        + (27054027 / 6029312) * np.cos(10 * phi + theta)
        + 572296725 / 486866944
    )


def M_gg_2_6_12_2():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_12_2(theta, phi):
    return (
        -307432125 / 229113856 * np.cos(4 * phi)
        - 1990989 / 1507328 * np.cos(8 * phi)
        + (2433431 / 524288) * np.cos(12 * phi)
        + (225450225 / 229113856) * np.cos(2 * theta)
        + (9018009 / 7159808) * np.cos(4 * theta)
        + (33066033 / 229113856) * np.cos(6 * theta)
        + (99198099 / 229113856) * np.cos(2 * phi - 5 * theta)
        + (12297285 / 7159808) * np.cos(2 * phi - 3 * theta)
        - 61486425 / 229113856 * np.cos(2 * phi - theta)
        - 20495475 / 28639232 * np.cos(2 * phi + theta)
        + (381215835 / 229113856) * np.cos(2 * phi + 3 * theta)
        + (18621603 / 28639232) * np.cos(2 * phi + 5 * theta)
        + (6441435 / 229113856) * np.cos(2 * phi + 7 * theta)
        + (225450225 / 229113856) * np.cos(4 * phi - 4 * theta)
        + (184459275 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (20495475 / 28639232) * np.cos(4 * phi + 2 * theta)
        + (342567225 / 229113856) * np.cos(4 * phi + 4 * theta)
        + (43918875 / 229113856) * np.cos(4 * phi + 6 * theta)
        + (425850425 / 229113856) * np.cos(6 * phi - 3 * theta)
        + (16591575 / 28639232) * np.cos(6 * phi - theta)
        - 282056775 / 229113856 * np.cos(6 * phi + theta)
        + (57675475 / 28639232) * np.cos(6 * phi + 3 * theta)
        + (82957875 / 114556928) * np.cos(6 * phi + 5 * theta)
        + (36501465 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (9954945 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (11851125 / 6029312) * np.cos(8 * phi + 4 * theta)
        + (51102051 / 12058624) * np.cos(10 * phi - theta)
        - 1990989 / 753664 * np.cos(10 * phi + theta)
        + (49774725 / 12058624) * np.cos(10 * phi + 3 * theta)
        + (3318315 / 524288) * np.cos(12 * phi + 2 * theta)
        - 75150075 / 114556928
    )


def M_gg_2_6_12_3():
    def func(k):
        return (8 / 422374095) * k**12

    return func


def N_gg_2_6_12_3(theta, phi):
    return (
        (70945875 / 28639232) * np.cos(4 * phi)
        + (34459425 / 12058624) * np.cos(8 * phi)
        + (36018675 / 28639232) * np.cos(2 * theta)
        + (36018675 / 229113856) * np.cos(4 * theta)
        + (6081075 / 114556928) * np.cos(2 * phi - 5 * theta)
        + (42567525 / 57278464) * np.cos(2 * phi - 3 * theta)
        + (127702575 / 57278464) * np.cos(2 * phi - theta)
        + (212837625 / 114556928) * np.cos(2 * phi + theta)
        + (42567525 / 114556928) * np.cos(2 * phi + 3 * theta)
        + (10135125 / 916455424) * np.cos(4 * phi - 6 * theta)
        + (10135125 / 28639232) * np.cos(4 * phi - 4 * theta)
        + (212837625 / 114556928) * np.cos(4 * phi - 2 * theta)
        + (354729375 / 458227712) * np.cos(4 * phi + 2 * theta)
        + (24613875 / 229113856) * np.cos(6 * phi - 5 * theta)
        + (73841625 / 57278464) * np.cos(6 * phi - 3 * theta)
        + (172297125 / 57278464) * np.cos(6 * phi - theta)
        + (172297125 / 114556928) * np.cos(6 * phi + theta)
        + (14768325 / 24117248) * np.cos(8 * phi - 4 * theta)
        + (4922775 / 1507328) * np.cos(8 * phi - 2 * theta)
        + (34459425 / 12058624) * np.cos(10 * phi - 3 * theta)
        + (34459425 / 6029312) * np.cos(10 * phi - theta)
        + (34459425 / 2097152) * np.cos(12 * phi - 2 * theta)
        + 540280125 / 458227712
    )


def M_gg_2_6_12_4():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_12_4(theta, phi):
    return (
        -307432125 / 229113856 * np.cos(4 * phi)
        - 1990989 / 1507328 * np.cos(8 * phi)
        + (2433431 / 524288) * np.cos(12 * phi)
        + (225450225 / 229113856) * np.cos(2 * theta)
        + (9018009 / 7159808) * np.cos(4 * theta)
        + (33066033 / 229113856) * np.cos(6 * theta)
        + (6441435 / 229113856) * np.cos(2 * phi - 7 * theta)
        + (18621603 / 28639232) * np.cos(2 * phi - 5 * theta)
        + (381215835 / 229113856) * np.cos(2 * phi - 3 * theta)
        - 20495475 / 28639232 * np.cos(2 * phi - theta)
        - 61486425 / 229113856 * np.cos(2 * phi + theta)
        + (12297285 / 7159808) * np.cos(2 * phi + 3 * theta)
        + (99198099 / 229113856) * np.cos(2 * phi + 5 * theta)
        + (43918875 / 229113856) * np.cos(4 * phi - 6 * theta)
        + (342567225 / 229113856) * np.cos(4 * phi - 4 * theta)
        + (20495475 / 28639232) * np.cos(4 * phi - 2 * theta)
        + (184459275 / 114556928) * np.cos(4 * phi + 2 * theta)
        + (225450225 / 229113856) * np.cos(4 * phi + 4 * theta)
        + (82957875 / 114556928) * np.cos(6 * phi - 5 * theta)
        + (57675475 / 28639232) * np.cos(6 * phi - 3 * theta)
        - 282056775 / 229113856 * np.cos(6 * phi - theta)
        + (16591575 / 28639232) * np.cos(6 * phi + theta)
        + (425850425 / 229113856) * np.cos(6 * phi + 3 * theta)
        + (11851125 / 6029312) * np.cos(8 * phi - 4 * theta)
        + (9954945 / 12058624) * np.cos(8 * phi - 2 * theta)
        + (36501465 / 12058624) * np.cos(8 * phi + 2 * theta)
        + (49774725 / 12058624) * np.cos(10 * phi - 3 * theta)
        - 1990989 / 753664 * np.cos(10 * phi - theta)
        + (51102051 / 12058624) * np.cos(10 * phi + theta)
        + (3318315 / 524288) * np.cos(12 * phi - 2 * theta)
        - 75150075 / 114556928
    )


def M_gg_2_6_12_5():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_12_5(theta, phi):
    return (
        (30956050125 / 26577207296) * np.cos(4 * phi)
        + (2205250047 / 1398800384) * np.cos(8 * phi)
        + (289578289 / 60817408) * np.cos(12 * phi)
        - 29686058325 / 26577207296 * np.cos(2 * theta)
        + (29546359227 / 26577207296) * np.cos(4 * theta)
        + (27284564307 / 26577207296) * np.cos(6 * theta)
        + (3243014775 / 53154414592) * np.cos(8 * theta)
        + (8431838415 / 26577207296) * np.cos(2 * phi - 7 * theta)
        + (44234721531 / 26577207296) * np.cos(2 * phi - 5 * theta)
        - 11144178045 / 26577207296 * np.cos(2 * phi - 3 * theta)
        + (2063736675 / 26577207296) * np.cos(2 * phi - theta)
        + (2063736675 / 26577207296) * np.cos(2 * phi + theta)
        - 11144178045 / 26577207296 * np.cos(2 * phi + 3 * theta)
        + (44234721531 / 26577207296) * np.cos(2 * phi + 5 * theta)
        + (8431838415 / 26577207296) * np.cos(2 * phi + 7 * theta)
        + (48645221625 / 53154414592) * np.cos(4 * phi - 6 * theta)
        + (38621357775 / 26577207296) * np.cos(4 * phi - 4 * theta)
        - 2063736675 / 1661075456 * np.cos(4 * phi - 2 * theta)
        - 2063736675 / 1661075456 * np.cos(4 * phi + 2 * theta)
        + (38621357775 / 26577207296) * np.cos(4 * phi + 4 * theta)
        + (48645221625 / 53154414592) * np.cos(4 * phi + 6 * theta)
        + (25059659625 / 13288603648) * np.cos(6 * phi - 5 * theta)
        + (556881325 / 26577207296) * np.cos(6 * phi - 3 * theta)
        - 1670643975 / 26577207296 * np.cos(6 * phi - theta)
        - 1670643975 / 26577207296 * np.cos(6 * phi + theta)
        + (556881325 / 26577207296) * np.cos(6 * phi + 3 * theta)
        + (25059659625 / 13288603648) * np.cos(6 * phi + 5 * theta)
        + (8353219875 / 2797600768) * np.cos(8 * phi - 4 * theta)
        - 2338901565 / 1398800384 * np.cos(8 * phi - 2 * theta)
        - 2338901565 / 1398800384 * np.cos(8 * phi + 2 * theta)
        + (8353219875 / 2797600768) * np.cos(8 * phi + 4 * theta)
        + (5011931925 / 1398800384) * np.cos(10 * phi - 3 * theta)
        - 1269689421 / 1398800384 * np.cos(10 * phi - theta)
        - 1269689421 / 1398800384 * np.cos(10 * phi + theta)
        + (5011931925 / 1398800384) * np.cos(10 * phi + 3 * theta)
        + (334128795 / 121634816) * np.cos(12 * phi - 2 * theta)
        + (334128795 / 121634816) * np.cos(12 * phi + 2 * theta)
        + 28521899175 / 53154414592
    )


def M_gg_2_6_14_0():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_14_0(theta, phi):
    return (
        (2299592295 / 916455424) * np.cos(4 * phi)
        + (153306153 / 48234496) * np.cos(8 * phi)
        + (12167155 / 2097152) * np.cos(12 * phi)
        + (1289575287 / 916455424) * np.cos(2 * theta)
        + (1289575287 / 4582277120) * np.cos(4 * theta)
        + (61408347 / 4582277120) * np.cos(6 * theta)
        + (99198099 / 1832910848) * np.cos(2 * phi - 5 * theta)
        + (297594297 / 458227712) * np.cos(2 * phi - 3 * theta)
        + (7439857425 / 3665821696) * np.cos(2 * phi - theta)
        + (495990495 / 229113856) * np.cos(2 * phi + theta)
        + (1487971485 / 1832910848) * np.cos(2 * phi + 3 * theta)
        + (42513471 / 458227712) * np.cos(2 * phi + 5 * theta)
        + (14171157 / 7331643392) * np.cos(2 * phi + 7 * theta)
        + (153306153 / 916455424) * np.cos(4 * phi - 4 * theta)
        + (2299592295 / 1832910848) * np.cos(4 * phi - 2 * theta)
        + (766530765 / 458227712) * np.cos(4 * phi + 2 * theta)
        + (328513185 / 916455424) * np.cos(4 * phi + 4 * theta)
        + (65702637 / 3665821696) * np.cos(4 * phi + 6 * theta)
        + (85170085 / 192937984) * np.cos(6 * phi - 3 * theta)
        + (51102051 / 24117248) * np.cos(6 * phi - theta)
        + (255510255 / 96468992) * np.cos(6 * phi + theta)
        + (12167155 / 12058624) * np.cos(6 * phi + 3 * theta)
        + (36501465 / 385875968) * np.cos(6 * phi + 5 * theta)
        + (51102051 / 48234496) * np.cos(8 * phi - 2 * theta)
        + (109504395 / 48234496) * np.cos(8 * phi + 2 * theta)
        + (36501465 / 96468992) * np.cos(8 * phi + 4 * theta)
        + (51102051 / 20971520) * np.cos(10 * phi - theta)
        + (21900879 / 5242880) * np.cos(10 * phi + theta)
        + (21900879 / 16777216) * np.cos(10 * phi + 3 * theta)
        + (36501465 / 8388608) * np.cos(12 * phi + 2 * theta)
        + (328513185 / 16777216) * np.cos(14 * phi + theta)
        + 2149292145 / 1832910848
    )


def M_gg_2_6_14_1():
    def func(k):
        return (32 / 6335611425) * k**12

    return func


def N_gg_2_6_14_1(theta, phi):
    return (
        (2299592295 / 916455424) * np.cos(4 * phi)
        + (153306153 / 48234496) * np.cos(8 * phi)
        + (12167155 / 2097152) * np.cos(12 * phi)
        + (1289575287 / 916455424) * np.cos(2 * theta)
        + (1289575287 / 4582277120) * np.cos(4 * theta)
        + (61408347 / 4582277120) * np.cos(6 * theta)
        + (14171157 / 7331643392) * np.cos(2 * phi - 7 * theta)
        + (42513471 / 458227712) * np.cos(2 * phi - 5 * theta)
        + (1487971485 / 1832910848) * np.cos(2 * phi - 3 * theta)
        + (495990495 / 229113856) * np.cos(2 * phi - theta)
        + (7439857425 / 3665821696) * np.cos(2 * phi + theta)
        + (297594297 / 458227712) * np.cos(2 * phi + 3 * theta)
        + (99198099 / 1832910848) * np.cos(2 * phi + 5 * theta)
        + (65702637 / 3665821696) * np.cos(4 * phi - 6 * theta)
        + (328513185 / 916455424) * np.cos(4 * phi - 4 * theta)
        + (766530765 / 458227712) * np.cos(4 * phi - 2 * theta)
        + (2299592295 / 1832910848) * np.cos(4 * phi + 2 * theta)
        + (153306153 / 916455424) * np.cos(4 * phi + 4 * theta)
        + (36501465 / 385875968) * np.cos(6 * phi - 5 * theta)
        + (12167155 / 12058624) * np.cos(6 * phi - 3 * theta)
        + (255510255 / 96468992) * np.cos(6 * phi - theta)
        + (51102051 / 24117248) * np.cos(6 * phi + theta)
        + (85170085 / 192937984) * np.cos(6 * phi + 3 * theta)
        + (36501465 / 96468992) * np.cos(8 * phi - 4 * theta)
        + (109504395 / 48234496) * np.cos(8 * phi - 2 * theta)
        + (51102051 / 48234496) * np.cos(8 * phi + 2 * theta)
        + (21900879 / 16777216) * np.cos(10 * phi - 3 * theta)
        + (21900879 / 5242880) * np.cos(10 * phi - theta)
        + (51102051 / 20971520) * np.cos(10 * phi + theta)
        + (36501465 / 8388608) * np.cos(12 * phi - 2 * theta)
        + (328513185 / 16777216) * np.cos(14 * phi - theta)
        + 2149292145 / 1832910848
    )


def M_gg_2_6_14_2():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_14_2(theta, phi):
    return (
        -39093069015 / 28410118144 * np.cos(4 * phi)
        - 2606204601 / 1495269376 * np.cos(8 * phi)
        - 206841635 / 65011712 * np.cos(12 * phi)
        + (21922779879 / 28410118144) * np.cos(2 * theta)
        + (197305018911 / 142050590720) * np.cos(4 * theta)
        + (39222388491 / 142050590720) * np.cos(6 * theta)
        + (447403671 / 56820236288) * np.cos(8 * theta)
        + (3131825697 / 56820236288) * np.cos(2 * phi - 7 * theta)
        + (21440960541 / 28410118144) * np.cos(2 * phi - 5 * theta)
        + (45531927441 / 28410118144) * np.cos(2 * phi - 3 * theta)
        - 8431838415 / 14205059072 * np.cos(2 * phi - theta)
        - 8431838415 / 14205059072 * np.cos(2 * phi + theta)
        + (45531927441 / 28410118144) * np.cos(2 * phi + 3 * theta)
        + (21440960541 / 28410118144) * np.cos(2 * phi + 5 * theta)
        + (3131825697 / 56820236288) * np.cos(2 * phi + 7 * theta)
        + (12286393119 / 56820236288) * np.cos(4 * phi - 6 * theta)
        + (40582328787 / 28410118144) * np.cos(4 * phi - 4 * theta)
        + (13031023005 / 14205059072) * np.cos(4 * phi - 2 * theta)
        + (13031023005 / 14205059072) * np.cos(4 * phi + 2 * theta)
        + (40582328787 / 28410118144) * np.cos(4 * phi + 4 * theta)
        + (12286393119 / 56820236288) * np.cos(4 * phi + 6 * theta)
        + (1861574715 / 2990538752) * np.cos(6 * phi - 5 * theta)
        + (1447891445 / 747634688) * np.cos(6 * phi - 3 * theta)
        - 868734867 / 1495269376 * np.cos(6 * phi - theta)
        - 868734867 / 1495269376 * np.cos(6 * phi + theta)
        + (1447891445 / 747634688) * np.cos(6 * phi + 3 * theta)
        + (1861574715 / 2990538752) * np.cos(6 * phi + 5 * theta)
        + (4343674335 / 2990538752) * np.cos(8 * phi - 4 * theta)
        + (2357994639 / 1495269376) * np.cos(8 * phi - 2 * theta)
        + (2357994639 / 1495269376) * np.cos(8 * phi + 2 * theta)
        + (4343674335 / 2990538752) * np.cos(8 * phi + 4 * theta)
        + (372314943 / 130023424) * np.cos(10 * phi - 3 * theta)
        - 124104981 / 325058560 * np.cos(10 * phi - theta)
        - 124104981 / 325058560 * np.cos(10 * phi + theta)
        + (372314943 / 130023424) * np.cos(10 * phi + 3 * theta)
        + (620524905 / 130023424) * np.cos(12 * phi - 2 * theta)
        + (620524905 / 130023424) * np.cos(12 * phi + 2 * theta)
        + (797817735 / 130023424) * np.cos(14 * phi - theta)
        + (797817735 / 130023424) * np.cos(14 * phi + theta)
        - 36537966465 / 56820236288
    )


def M_gg_2_6_16_0():
    def func(k):
        return (64 / 107705394225) * k**12

    return func


def N_gg_2_6_16_0(theta, phi):
    return (
        (430023759165 / 173451247616) * np.cos(4 * phi)
        + (91217161035 / 30165434368) * np.cos(8 * phi)
        + (35901798075 / 7541358592) * np.cos(12 * phi)
        + (11967266025 / 536870912) * np.cos(16 * phi)
        + (4932625472775 / 3295573704704) * np.cos(2 * theta)
        + (4932625472775 / 13182294818816) * np.cos(4 * theta)
        + (100665825975 / 3295573704704) * np.cos(6 * theta)
        + (100665825975 / 210916717101056) * np.cos(8 * theta)
        + (114087936105 / 26364589637632) * np.cos(2 * phi - 7 * theta)
        + (798615552735 / 6591147409408) * np.cos(2 * phi - 5 * theta)
        + (5590308869145 / 6591147409408) * np.cos(2 * phi - 3 * theta)
        + (27951544345725 / 13182294818816) * np.cos(2 * phi - theta)
        + (27951544345725 / 13182294818816) * np.cos(2 * phi + theta)
        + (5590308869145 / 6591147409408) * np.cos(2 * phi + 3 * theta)
        + (798615552735 / 6591147409408) * np.cos(2 * phi + 5 * theta)
        + (114087936105 / 26364589637632) * np.cos(2 * phi + 7 * theta)
        + (61431965595 / 2775219961856) * np.cos(4 * phi - 6 * theta)
        + (61431965595 / 173451247616) * np.cos(4 * phi - 4 * theta)
        + (2150118795825 / 1387609980928) * np.cos(4 * phi - 2 * theta)
        + (2150118795825 / 1387609980928) * np.cos(4 * phi + 2 * theta)
        + (61431965595 / 173451247616) * np.cos(4 * phi + 4 * theta)
        + (61431965595 / 2775219961856) * np.cos(4 * phi + 6 * theta)
        + (117279207045 / 1387609980928) * np.cos(6 * phi - 5 * theta)
        + (586396035225 / 693804990464) * np.cos(6 * phi - 3 * theta)
        + (820954449315 / 346902495232) * np.cos(6 * phi - theta)
        + (820954449315 / 346902495232) * np.cos(6 * phi + theta)
        + (586396035225 / 693804990464) * np.cos(6 * phi + 3 * theta)
        + (117279207045 / 1387609980928) * np.cos(6 * phi + 5 * theta)
        + (65155115025 / 241323474944) * np.cos(8 * phi - 4 * theta)
        + (13031023005 / 7541358592) * np.cos(8 * phi - 2 * theta)
        + (13031023005 / 7541358592) * np.cos(8 * phi + 2 * theta)
        + (65155115025 / 241323474944) * np.cos(8 * phi + 4 * theta)
        + (46539367875 / 60330868736) * np.cos(10 * phi - 3 * theta)
        + (46539367875 / 15082717184) * np.cos(10 * phi - theta)
        + (46539367875 / 15082717184) * np.cos(10 * phi + theta)
        + (46539367875 / 60330868736) * np.cos(10 * phi + 3 * theta)
        + (251312586525 / 120661737472) * np.cos(12 * phi - 2 * theta)
        + (251312586525 / 120661737472) * np.cos(12 * phi + 2 * theta)
        + (11967266025 / 2080374784) * np.cos(14 * phi - theta)
        + (11967266025 / 2080374784) * np.cos(14 * phi + theta)
        + 123315636819375 / 105458358550528
    )


def M_gv_0_0_1_0():
    def func(k):
        return (100 / 3) / k

    return func


def N_gv_0_0_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_0_1_1_0():
    def func(k):
        return -50 / 9 * k

    return func


def N_gv_0_1_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_0_1_1_1():
    def func(k):
        return -20 / 9 * k

    return func


def N_gv_0_1_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_0_1_3_0():
    def func(k):
        return -20 / 9 * k

    return func


def N_gv_0_1_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_0_2_1_0():
    def func(k):
        return (5 / 6) * k**3

    return func


def N_gv_0_2_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_0_2_1_1():
    def func(k):
        return (10 / 21) * k**3

    return func


def N_gv_0_2_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_0_2_3_0():
    def func(k):
        return (10 / 21) * k**3

    return func


def N_gv_0_2_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_0_2_3_1():
    def func(k):
        return (20 / 189) * k**3

    return func


def N_gv_0_2_3_1(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_0_2_5_0():
    def func(k):
        return (20 / 189) * k**3

    return func


def N_gv_0_2_5_0(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_0_3_1_0():
    def func(k):
        return -25 / 252 * k**5

    return func


def N_gv_0_3_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_0_3_1_1():
    def func(k):
        return -25 / 378 * k**5

    return func


def N_gv_0_3_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_0_3_3_0():
    def func(k):
        return -25 / 378 * k**5

    return func


def N_gv_0_3_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_0_3_3_1():
    def func(k):
        return -50 / 2079 * k**5

    return func


def N_gv_0_3_3_1(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_0_3_5_0():
    def func(k):
        return -50 / 2079 * k**5

    return func


def N_gv_0_3_5_0(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_0_3_5_1():
    def func(k):
        return -100 / 27027 * k**5

    return func


def N_gv_0_3_5_1(theta, phi):
    return (
        -315 / 128 * np.cos(phi - 3 / 2 * theta)
        - 225 / 128 * np.cos(phi + (1 / 2) * theta)
        - 945 / 256 * np.cos(3 * phi - 5 / 2 * theta)
        - 315 / 256 * np.cos(3 * phi - 1 / 2 * theta)
        - 2079 / 256 * np.cos(5 * phi - 7 / 2 * theta)
        - 189 / 256 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_0_3_7_0():
    def func(k):
        return -100 / 27027 * k**5

    return func


def N_gv_0_3_7_0(theta, phi):
    return (
        -1575 / 1024 * np.cos(phi - 3 / 2 * theta)
        - 525 / 256 * np.cos(phi + (1 / 2) * theta)
        - 567 / 512 * np.cos(3 * phi - 5 / 2 * theta)
        - 2835 / 1024 * np.cos(3 * phi - 1 / 2 * theta)
        - 693 / 1024 * np.cos(5 * phi - 7 / 2 * theta)
        - 2079 / 512 * np.cos(5 * phi - 3 / 2 * theta)
        - 9009 / 1024 * np.cos(7 * phi - 5 / 2 * theta)
    )


def M_gv_1_0_1_0():
    def func(k):
        return (100 / 9) / k

    return func


def N_gv_1_0_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_0_1_1():
    def func(k):
        return (40 / 9) / k

    return func


def N_gv_1_0_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_0_3_0():
    def func(k):
        return (40 / 9) / k

    return func


def N_gv_1_0_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_1_1_1_0():
    def func(k):
        return -10 / 3 * k

    return func


def N_gv_1_1_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_1_1_1():
    def func(k):
        return -40 / 21 * k

    return func


def N_gv_1_1_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_1_3_0():
    def func(k):
        return -40 / 21 * k

    return func


def N_gv_1_1_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_1_1_3_1():
    def func(k):
        return -80 / 189 * k

    return func


def N_gv_1_1_3_1(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_1_1_5_0():
    def func(k):
        return -80 / 189 * k

    return func


def N_gv_1_1_5_0(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_1_2_1_0():
    def func(k):
        return (25 / 42) * k**3

    return func


def N_gv_1_2_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_2_1_1():
    def func(k):
        return (25 / 63) * k**3

    return func


def N_gv_1_2_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_2_3_0():
    def func(k):
        return (25 / 63) * k**3

    return func


def N_gv_1_2_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_1_2_3_1():
    def func(k):
        return (100 / 693) * k**3

    return func


def N_gv_1_2_3_1(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_1_2_5_0():
    def func(k):
        return (100 / 693) * k**3

    return func


def N_gv_1_2_5_0(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_1_2_5_1():
    def func(k):
        return (200 / 9009) * k**3

    return func


def N_gv_1_2_5_1(theta, phi):
    return (
        -315 / 128 * np.cos(phi - 3 / 2 * theta)
        - 225 / 128 * np.cos(phi + (1 / 2) * theta)
        - 945 / 256 * np.cos(3 * phi - 5 / 2 * theta)
        - 315 / 256 * np.cos(3 * phi - 1 / 2 * theta)
        - 2079 / 256 * np.cos(5 * phi - 7 / 2 * theta)
        - 189 / 256 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_1_2_7_0():
    def func(k):
        return (200 / 9009) * k**3

    return func


def N_gv_1_2_7_0(theta, phi):
    return (
        -1575 / 1024 * np.cos(phi - 3 / 2 * theta)
        - 525 / 256 * np.cos(phi + (1 / 2) * theta)
        - 567 / 512 * np.cos(3 * phi - 5 / 2 * theta)
        - 2835 / 1024 * np.cos(3 * phi - 1 / 2 * theta)
        - 693 / 1024 * np.cos(5 * phi - 7 / 2 * theta)
        - 2079 / 512 * np.cos(5 * phi - 3 / 2 * theta)
        - 9009 / 1024 * np.cos(7 * phi - 5 / 2 * theta)
    )


def M_gv_1_3_1_0():
    def func(k):
        return -25 / 324 * k**5

    return func


def N_gv_1_3_1_0(theta, phi):
    return -3 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_3_1_1():
    def func(k):
        return -50 / 891 * k**5

    return func


def N_gv_1_3_1_1(theta, phi):
    return -9 / 2 * np.cos(phi - 3 / 2 * theta) - 3 / 2 * np.cos(phi + (1 / 2) * theta)


def M_gv_1_3_3_0():
    def func(k):
        return -50 / 891 * k**5

    return func


def N_gv_1_3_3_0(theta, phi):
    return (
        -9 / 8 * np.cos(phi - 3 / 2 * theta)
        - 9 / 4 * np.cos(phi + (1 / 2) * theta)
        - 45 / 8 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_1_3_3_1():
    def func(k):
        return -100 / 3861 * k**5

    return func


def N_gv_1_3_3_1(theta, phi):
    return (
        -45 / 16 * np.cos(phi - 3 / 2 * theta)
        - 27 / 16 * np.cos(phi + (1 / 2) * theta)
        - 105 / 16 * np.cos(3 * phi - 5 / 2 * theta)
        - 15 / 16 * np.cos(3 * phi - 1 / 2 * theta)
    )


def M_gv_1_3_5_0():
    def func(k):
        return -100 / 3861 * k**5

    return func


def N_gv_1_3_5_0(theta, phi):
    return (
        -45 / 32 * np.cos(phi - 3 / 2 * theta)
        - 135 / 64 * np.cos(phi + (1 / 2) * theta)
        - 105 / 128 * np.cos(3 * phi - 5 / 2 * theta)
        - 105 / 32 * np.cos(3 * phi - 1 / 2 * theta)
        - 945 / 128 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_1_3_5_1():
    def func(k):
        return -80 / 11583 * k**5

    return func


def N_gv_1_3_5_1(theta, phi):
    return (
        -315 / 128 * np.cos(phi - 3 / 2 * theta)
        - 225 / 128 * np.cos(phi + (1 / 2) * theta)
        - 945 / 256 * np.cos(3 * phi - 5 / 2 * theta)
        - 315 / 256 * np.cos(3 * phi - 1 / 2 * theta)
        - 2079 / 256 * np.cos(5 * phi - 7 / 2 * theta)
        - 189 / 256 * np.cos(5 * phi - 3 / 2 * theta)
    )


def M_gv_1_3_7_0():
    def func(k):
        return -80 / 11583 * k**5

    return func


def N_gv_1_3_7_0(theta, phi):
    return (
        -1575 / 1024 * np.cos(phi - 3 / 2 * theta)
        - 525 / 256 * np.cos(phi + (1 / 2) * theta)
        - 567 / 512 * np.cos(3 * phi - 5 / 2 * theta)
        - 2835 / 1024 * np.cos(3 * phi - 1 / 2 * theta)
        - 693 / 1024 * np.cos(5 * phi - 7 / 2 * theta)
        - 2079 / 512 * np.cos(5 * phi - 3 / 2 * theta)
        - 9009 / 1024 * np.cos(7 * phi - 5 / 2 * theta)
    )


def M_gv_1_3_7_1():
    def func(k):
        return -160 / 196911 * k**5

    return func


def N_gv_1_3_7_1(theta, phi):
    return (
        -4725 / 2048 * np.cos(phi - 3 / 2 * theta)
        - 3675 / 2048 * np.cos(phi + (1 / 2) * theta)
        - 6237 / 2048 * np.cos(3 * phi - 5 / 2 * theta)
        - 2835 / 2048 * np.cos(3 * phi - 1 / 2 * theta)
        - 9009 / 2048 * np.cos(5 * phi - 7 / 2 * theta)
        - 2079 / 2048 * np.cos(5 * phi - 3 / 2 * theta)
        - 19305 / 2048 * np.cos(7 * phi - 9 / 2 * theta)
        - 1287 / 2048 * np.cos(7 * phi - 5 / 2 * theta)
    )


def M_gv_1_3_9_0():
    def func(k):
        return -160 / 196911 * k**5

    return func


def N_gv_1_3_9_0(theta, phi):
    return (
        -6615 / 4096 * np.cos(phi - 3 / 2 * theta)
        - 33075 / 16384 * np.cos(phi + (1 / 2) * theta)
        - 10395 / 8192 * np.cos(3 * phi - 5 / 2 * theta)
        - 10395 / 4096 * np.cos(3 * phi - 1 / 2 * theta)
        - 3861 / 4096 * np.cos(5 * phi - 7 / 2 * theta)
        - 27027 / 8192 * np.cos(5 * phi - 3 / 2 * theta)
        - 19305 / 32768 * np.cos(7 * phi - 9 / 2 * theta)
        - 19305 / 4096 * np.cos(7 * phi - 5 / 2 * theta)
        - 328185 / 32768 * np.cos(9 * phi - 7 / 2 * theta)
    )


def M_vv_0_0_0_0():
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_0_0_0(theta, phi):
    return 3 * np.cos(theta)


def M_vv_0_0_2_0():
    def func(k):
        return (10000 / 9) / k**2

    return func


def N_vv_0_0_2_0(theta, phi):
    return (9 / 2) * np.cos(2 * phi) + (3 / 2) * np.cos(theta)


dictionary_terms = {
    "gg": [
        "0_0",
        "0_1",
        "0_2",
        "0_3",
        "0_4",
        "0_5",
        "0_6",
        "1_0",
        "1_1",
        "1_2",
        "1_3",
        "1_4",
        "1_5",
        "1_6",
        "2_0",
        "2_1",
        "2_2",
        "2_3",
        "2_4",
        "2_5",
        "2_6",
    ],
    "gv": ["0_0", "0_1", "0_2", "0_3", "1_0", "1_1", "1_2", "1_3"],
    "vv": ["0_0"],
}
dictionary_lmax = {
    "gg": [0, 2, 4, 6, 8, 10, 12, 2, 4, 6, 8, 10, 12, 14, 4, 6, 8, 10, 12, 14, 16],
    "gv": [1, 3, 5, 7, 3, 5, 7, 9],
    "vv": [2],
}
dictionary_subterms = {
    "gg_0_0_0": 1,
    "gg_0_1_0": 1,
    "gg_0_1_1": 0,
    "gg_0_1_2": 2,
    "gg_0_2_0": 2,
    "gg_0_2_1": 0,
    "gg_0_2_2": 3,
    "gg_0_2_3": 0,
    "gg_0_2_4": 3,
    "gg_0_3_0": 2,
    "gg_0_3_1": 0,
    "gg_0_3_2": 5,
    "gg_0_3_3": 0,
    "gg_0_3_4": 5,
    "gg_0_3_5": 0,
    "gg_0_3_6": 4,
    "gg_0_4_0": 3,
    "gg_0_4_1": 0,
    "gg_0_4_2": 6,
    "gg_0_4_3": 0,
    "gg_0_4_4": 8,
    "gg_0_4_5": 0,
    "gg_0_4_6": 7,
    "gg_0_4_7": 0,
    "gg_0_4_8": 3,
    "gg_0_5_0": 3,
    "gg_0_5_1": 0,
    "gg_0_5_2": 8,
    "gg_0_5_3": 0,
    "gg_0_5_4": 10,
    "gg_0_5_5": 0,
    "gg_0_5_6": 9,
    "gg_0_5_7": 0,
    "gg_0_5_8": 5,
    "gg_0_5_9": 0,
    "gg_0_5_10": 2,
    "gg_0_6_0": 4,
    "gg_0_6_1": 0,
    "gg_0_6_2": 9,
    "gg_0_6_3": 0,
    "gg_0_6_4": 11,
    "gg_0_6_5": 0,
    "gg_0_6_6": 10,
    "gg_0_6_7": 0,
    "gg_0_6_8": 6,
    "gg_0_6_9": 0,
    "gg_0_6_10": 3,
    "gg_0_6_11": 0,
    "gg_0_6_12": 1,
    "gg_1_0_0": 1,
    "gg_1_0_1": 0,
    "gg_1_0_2": 2,
    "gg_1_1_0": 2,
    "gg_1_1_1": 0,
    "gg_1_1_2": 3,
    "gg_1_1_3": 0,
    "gg_1_1_4": 3,
    "gg_1_2_0": 2,
    "gg_1_2_1": 0,
    "gg_1_2_2": 5,
    "gg_1_2_3": 0,
    "gg_1_2_4": 5,
    "gg_1_2_5": 0,
    "gg_1_2_6": 4,
    "gg_1_3_0": 3,
    "gg_1_3_1": 0,
    "gg_1_3_2": 6,
    "gg_1_3_3": 0,
    "gg_1_3_4": 8,
    "gg_1_3_5": 0,
    "gg_1_3_6": 7,
    "gg_1_3_7": 0,
    "gg_1_3_8": 5,
    "gg_1_4_0": 3,
    "gg_1_4_1": 0,
    "gg_1_4_2": 8,
    "gg_1_4_3": 0,
    "gg_1_4_4": 10,
    "gg_1_4_5": 0,
    "gg_1_4_6": 11,
    "gg_1_4_7": 0,
    "gg_1_4_8": 9,
    "gg_1_4_9": 0,
    "gg_1_4_10": 4,
    "gg_1_5_0": 4,
    "gg_1_5_1": 0,
    "gg_1_5_2": 9,
    "gg_1_5_3": 0,
    "gg_1_5_4": 13,
    "gg_1_5_5": 0,
    "gg_1_5_6": 14,
    "gg_1_5_7": 0,
    "gg_1_5_8": 12,
    "gg_1_5_9": 0,
    "gg_1_5_10": 7,
    "gg_1_5_11": 0,
    "gg_1_5_12": 3,
    "gg_1_6_0": 4,
    "gg_1_6_1": 0,
    "gg_1_6_2": 11,
    "gg_1_6_3": 0,
    "gg_1_6_4": 15,
    "gg_1_6_5": 0,
    "gg_1_6_6": 16,
    "gg_1_6_7": 0,
    "gg_1_6_8": 14,
    "gg_1_6_9": 0,
    "gg_1_6_10": 9,
    "gg_1_6_11": 0,
    "gg_1_6_12": 5,
    "gg_1_6_13": 0,
    "gg_1_6_14": 2,
    "gg_2_0_0": 2,
    "gg_2_0_1": 0,
    "gg_2_0_2": 3,
    "gg_2_0_3": 0,
    "gg_2_0_4": 1,
    "gg_2_1_0": 2,
    "gg_2_1_1": 0,
    "gg_2_1_2": 5,
    "gg_2_1_3": 0,
    "gg_2_1_4": 5,
    "gg_2_1_5": 0,
    "gg_2_1_6": 2,
    "gg_2_2_0": 3,
    "gg_2_2_1": 0,
    "gg_2_2_2": 6,
    "gg_2_2_3": 0,
    "gg_2_2_4": 8,
    "gg_2_2_5": 0,
    "gg_2_2_6": 7,
    "gg_2_2_7": 0,
    "gg_2_2_8": 3,
    "gg_2_3_0": 3,
    "gg_2_3_1": 0,
    "gg_2_3_2": 8,
    "gg_2_3_3": 0,
    "gg_2_3_4": 10,
    "gg_2_3_5": 0,
    "gg_2_3_6": 11,
    "gg_2_3_7": 0,
    "gg_2_3_8": 9,
    "gg_2_3_9": 0,
    "gg_2_3_10": 4,
    "gg_2_4_0": 4,
    "gg_2_4_1": 0,
    "gg_2_4_2": 9,
    "gg_2_4_3": 0,
    "gg_2_4_4": 13,
    "gg_2_4_5": 0,
    "gg_2_4_6": 14,
    "gg_2_4_7": 0,
    "gg_2_4_8": 12,
    "gg_2_4_9": 0,
    "gg_2_4_10": 7,
    "gg_2_4_11": 0,
    "gg_2_4_12": 3,
    "gg_2_5_0": 4,
    "gg_2_5_1": 0,
    "gg_2_5_2": 11,
    "gg_2_5_3": 0,
    "gg_2_5_4": 15,
    "gg_2_5_5": 0,
    "gg_2_5_6": 16,
    "gg_2_5_7": 0,
    "gg_2_5_8": 14,
    "gg_2_5_9": 0,
    "gg_2_5_10": 9,
    "gg_2_5_11": 0,
    "gg_2_5_12": 5,
    "gg_2_5_13": 0,
    "gg_2_5_14": 2,
    "gg_2_6_0": 5,
    "gg_2_6_1": 0,
    "gg_2_6_2": 12,
    "gg_2_6_3": 0,
    "gg_2_6_4": 16,
    "gg_2_6_5": 0,
    "gg_2_6_6": 17,
    "gg_2_6_7": 0,
    "gg_2_6_8": 15,
    "gg_2_6_9": 0,
    "gg_2_6_10": 10,
    "gg_2_6_11": 0,
    "gg_2_6_12": 6,
    "gg_2_6_13": 0,
    "gg_2_6_14": 3,
    "gg_2_6_15": 0,
    "gg_2_6_16": 1,
    "gv_0_0_0": 0,
    "gv_0_0_1": 1,
    "gv_0_1_0": 0,
    "gv_0_1_1": 2,
    "gv_0_1_2": 0,
    "gv_0_1_3": 1,
    "gv_0_2_0": 0,
    "gv_0_2_1": 2,
    "gv_0_2_2": 0,
    "gv_0_2_3": 2,
    "gv_0_2_4": 0,
    "gv_0_2_5": 1,
    "gv_0_3_0": 0,
    "gv_0_3_1": 2,
    "gv_0_3_2": 0,
    "gv_0_3_3": 2,
    "gv_0_3_4": 0,
    "gv_0_3_5": 2,
    "gv_0_3_6": 0,
    "gv_0_3_7": 1,
    "gv_1_0_0": 0,
    "gv_1_0_1": 2,
    "gv_1_0_2": 0,
    "gv_1_0_3": 1,
    "gv_1_1_0": 0,
    "gv_1_1_1": 2,
    "gv_1_1_2": 0,
    "gv_1_1_3": 2,
    "gv_1_1_4": 0,
    "gv_1_1_5": 1,
    "gv_1_2_0": 0,
    "gv_1_2_1": 2,
    "gv_1_2_2": 0,
    "gv_1_2_3": 2,
    "gv_1_2_4": 0,
    "gv_1_2_5": 2,
    "gv_1_2_6": 0,
    "gv_1_2_7": 1,
    "gv_1_3_0": 0,
    "gv_1_3_1": 2,
    "gv_1_3_2": 0,
    "gv_1_3_3": 2,
    "gv_1_3_4": 0,
    "gv_1_3_5": 2,
    "gv_1_3_6": 0,
    "gv_1_3_7": 2,
    "gv_1_3_8": 0,
    "gv_1_3_9": 1,
    "vv_0_0_0": 1,
    "vv_0_0_1": 0,
    "vv_0_0_2": 1,
}
multi_index_model = True
regularize_M_terms = None
