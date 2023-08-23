import numpy as np


def K_gg_0_0(theta, phi, sig_g):
    def func(k):
        return (
            (1 / 2)
            * np.pi
            * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**2 * sig_g**2)
        )

    return func


def K_gg_0_1(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_0_2(theta, phi, sig_g):
    def func(k):
        return (
            (15 / 224)
            * (
                np.pi
                * k**4
                * sig_g**4
                * (
                    5 * np.cos(2 * theta)
                    - 9 * np.cos(2 * phi - theta)
                    - 9 * np.cos(2 * phi + theta)
                    - 11
                )
                * np.exp(2 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 2
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**3
                * sig_g**3
                * (
                    15 * np.cos(2 * theta)
                    - 6 * np.cos(2 * phi - theta)
                    - 6 * np.cos(2 * phi + theta)
                    - 19
                )
                * np.exp((3 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 2
                * np.pi
                * k**2
                * sig_g**2
                * (
                    -15 * np.cos(2 * theta)
                    + 6 * np.cos(2 * phi - theta)
                    + 6 * np.cos(2 * phi + theta)
                    + 19
                )
                * np.exp(2 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 30
                * k**2
                * sig_g**2
                * (
                    3 * np.cos(2 * theta)
                    + 3 * np.cos(2 * phi - theta)
                    + 3 * np.cos(2 * phi + theta)
                    - 1
                )
                * np.exp(k**2 * sig_g**2)
                + 30
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k
                * sig_g
                * (
                    -3 * np.cos(2 * theta)
                    - 3 * np.cos(2 * phi - theta)
                    - 3 * np.cos(2 * phi + theta)
                    + 1
                )
                * np.exp((3 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 15
                * np.pi
                * (
                    3 * np.cos(2 * theta)
                    + 3 * np.cos(2 * phi - theta)
                    + 3 * np.cos(2 * phi + theta)
                    - 1
                )
                * np.exp(2 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            )
            * np.exp(-2 * k**2 * sig_g**2)
            / (k**6 * sig_g**6)
        )

    return func


def K_gg_0_3(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_0_4(theta, phi, sig_g):
    def func(k):
        return (
            (45 / 4685824)
            * (
                np.pi
                * k**8
                * sig_g**8
                * (
                    4822335 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 9199260 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 11553430 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 1908760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 1055215 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 4822335 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 10090080 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 3576545 * np.sin(phi) ** 4
                    - 2041200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 2000880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 10261160 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6258760 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 2110430 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 2041200 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 6567120 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 2593120 * np.sin(phi) ** 2
                    - 449064 * np.sin((1 / 2) * theta) ** 8
                    + 2348640 * np.sin((1 / 2) * theta) ** 6
                    + 3373608 * np.sin((1 / 2) * theta) ** 4
                    - 6271288 * np.sin((1 / 2) * theta) ** 2
                    + 1217958 * np.sin(theta) ** 2
                    + 2408616 * np.cos((1 / 2) * theta) ** 8
                    - 3165120 * np.cos((1 / 2) * theta) ** 6
                    + 1143064
                )
                * np.exp(5 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 6
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**7
                * sig_g**7
                * (
                    -2679075 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 4703020 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 872830 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 4485320 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 727405 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 2679075 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 6013280 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 3184125 * np.sin(phi) ** 4
                    + 1134000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 1868720 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 4400760 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 8713560 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 1454810 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 1134000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 4754960 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 3449440 * np.sin(phi) ** 2
                    + 249480 * np.sin((1 / 2) * theta) ** 8
                    - 838880 * np.sin((1 / 2) * theta) ** 6
                    + 3732344 * np.sin((1 / 2) * theta) ** 4
                    - 3167144 * np.sin((1 / 2) * theta) ** 2
                    + 804594 * np.sin(theta) ** 2
                    - 1338120 * np.cos((1 / 2) * theta) ** 8
                    + 1292480 * np.cos((1 / 2) * theta) ** 6
                    + 11336
                )
                * np.exp((9 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 12
                * np.pi
                * k**6
                * sig_g**6
                * (
                    -8037225 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 15026340 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 7408730 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 8665720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 773465 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 8037225 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 17122560 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 7819735 * np.sin(phi) ** 4
                    + 3402000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 3902640 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 1838200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6673960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 1546930 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 3402000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 11775120 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 6926720 * np.sin(phi) ** 2
                    + 748440 * np.sin((1 / 2) * theta) ** 8
                    - 3564960 * np.sin((1 / 2) * theta) ** 6
                    - 456792 * np.sin((1 / 2) * theta) ** 4
                    + 4365512 * np.sin((1 / 2) * theta) ** 2
                    - 919002 * np.sin(theta) ** 2
                    - 4014360 * np.cos((1 / 2) * theta) ** 8
                    + 4925760 * np.cos((1 / 2) * theta) ** 6
                    - 1200680
                )
                * np.exp(5 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 42
                * k**6
                * sig_g**6
                * (
                    637875 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 1022700 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 1924510 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 648760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 273315 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 637875 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 1528800 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 283885 * np.sin(phi) ** 4
                    - 270000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 625200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 2621320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 1313960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 546630 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 270000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 1395600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 431840 * np.sin(phi) ** 2
                    - 59400 * np.sin((1 / 2) * theta) ** 8
                    + 88800 * np.sin((1 / 2) * theta) ** 6
                    + 1225736 * np.sin((1 / 2) * theta) ** 4
                    - 1220056 * np.sin((1 / 2) * theta) ** 2
                    + 301806 * np.sin(theta) ** 2
                    + 318600 * np.cos((1 / 2) * theta) ** 8
                    - 196800 * np.cos((1 / 2) * theta) ** 6
                    + 16952
                )
                * np.exp(4 * k**2 * sig_g**2)
                + 42
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**5
                * sig_g**5
                * (
                    -4209975 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 8322300 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 6563270 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 1856680 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 720615 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 4209975 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 8517600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 3772265 * np.sin(phi) ** 4
                    + 1782000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 1206000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 4133480 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 1701640 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 1441230 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 1782000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 4942800 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 2548960 * np.sin(phi) ** 2
                    + 392040 * np.sin((1 / 2) * theta) ** 8
                    - 2383200 * np.sin((1 / 2) * theta) ** 6
                    - 1409896 * np.sin((1 / 2) * theta) ** 4
                    + 4470776 * np.sin((1 / 2) * theta) ** 2
                    - 852246 * np.sin(theta) ** 2
                    - 2102760 * np.cos((1 / 2) * theta) ** 8
                    + 3096000 * np.cos((1 / 2) * theta) ** 6
                    - 1115608
                )
                * np.exp((9 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 1260
                * k**4
                * sig_g**4
                * (
                    893025 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 1519140 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 417130 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 1368920 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 19215 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 893025 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 2052960 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 915775 * np.sin(phi) ** 4
                    - 378000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 713040 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 364520 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 1968520 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 38430 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 378000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 1716720 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 1059680 * np.sin(phi) ** 2
                    - 83160 * np.sin((1 / 2) * theta) ** 8
                    + 224160 * np.sin((1 / 2) * theta) ** 6
                    - 278440 * np.sin((1 / 2) * theta) ** 4
                    + 173240 * np.sin((1 / 2) * theta) ** 2
                    - 21510 * np.sin(theta) ** 2
                    + 446040 * np.cos((1 / 2) * theta) ** 8
                    - 375360 * np.cos((1 / 2) * theta) ** 6
                    - 14872
                )
                * np.exp(4 * k**2 * sig_g**2)
                + 126
                * np.pi
                * k**4
                * sig_g**4
                * (
                    6506325 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 11974900 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 1266930 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 11745720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 300965 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 6506325 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 14050400 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 7013195 * np.sin(phi) ** 4
                    - 2754000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 3510800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 4218760 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 12115880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 601930 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 2754000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 10046000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 6685280 * np.sin(phi) ** 2
                    - 605880 * np.sin((1 / 2) * theta) ** 8
                    + 2669600 * np.sin((1 / 2) * theta) ** 6
                    - 1351624 * np.sin((1 / 2) * theta) ** 4
                    - 1309416 * np.sin((1 / 2) * theta) ** 2
                    + 373506 * np.sin(theta) ** 2
                    + 3249720 * np.cos((1 / 2) * theta) ** 8
                    - 3771200 * np.cos((1 / 2) * theta) ** 6
                    + 642824
                )
                * np.exp(5 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 3150
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**3
                * sig_g**3
                * (
                    893025 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 1641444 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 310870 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 2096920 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 33201 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 893025 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 1930656 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 1026431 * np.sin(phi) ** 4
                    - 378000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 485904 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 1092520 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 2203976 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 66402 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 378000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 1384752 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 993952 * np.sin(phi) ** 2
                    - 83160 * np.sin((1 / 2) * theta) ** 8
                    + 363936 * np.sin((1 / 2) * theta) ** 6
                    - 276776 * np.sin((1 / 2) * theta) ** 4
                    - 74696 * np.sin((1 / 2) * theta) ** 2
                    + 42138 * np.sin(theta) ** 2
                    + 446040 * np.cos((1 / 2) * theta) ** 8
                    - 515136 * np.cos((1 / 2) * theta) ** 6
                    + 71656
                )
                * np.exp((9 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 1260
                * np.pi
                * k**2
                * sig_g**2
                * (
                    -2679075 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 4863180 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 568610 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 5926760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 73395 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 2679075 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 5853120 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 3023965 * np.sin(phi) ** 4
                    + 1134000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 1571280 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 2913560 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6494200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 146790 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 1134000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 4320240 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 3014720 * np.sin(phi) ** 2
                    + 249480 * np.sin((1 / 2) * theta) ** 8
                    - 1021920 * np.sin((1 / 2) * theta) ** 6
                    + 831160 * np.sin((1 / 2) * theta) ** 4
                    + 100120 * np.sin((1 / 2) * theta) ** 2
                    - 94590 * np.sin(theta) ** 2
                    - 1338120 * np.cos((1 / 2) * theta) ** 8
                    + 1475520 * np.cos((1 / 2) * theta) ** 6
                    - 171704
                )
                * np.exp(5 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 51030
                * k**2
                * sig_g**2
                * (
                    231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 257915 * np.sin(phi) ** 4
                    - 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 262560 * np.sin(phi) ** 2
                    - 21560 * np.sin((1 / 2) * theta) ** 8
                    + 84000 * np.sin((1 / 2) * theta) ** 6
                    - 71880 * np.sin((1 / 2) * theta) ** 4
                    - 1000 * np.sin((1 / 2) * theta) ** 2
                    + 6210 * np.sin(theta) ** 2
                    + 115640 * np.cos((1 / 2) * theta) ** 8
                    - 123200 * np.cos((1 / 2) * theta) ** 6
                    + 12168
                )
                * np.exp(4 * k**2 * sig_g**2)
                + 51030
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k
                * sig_g
                * (
                    -231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 257915 * np.sin(phi) ** 4
                    + 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 262560 * np.sin(phi) ** 2
                    + 21560 * np.sin((1 / 2) * theta) ** 8
                    - 84000 * np.sin((1 / 2) * theta) ** 6
                    + 71880 * np.sin((1 / 2) * theta) ** 4
                    + 1000 * np.sin((1 / 2) * theta) ** 2
                    - 6210 * np.sin(theta) ** 2
                    - 115640 * np.cos((1 / 2) * theta) ** 8
                    + 123200 * np.cos((1 / 2) * theta) ** 6
                    - 12168
                )
                * np.exp((9 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 25515
                * np.pi
                * (
                    231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 257915 * np.sin(phi) ** 4
                    - 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 262560 * np.sin(phi) ** 2
                    - 21560 * np.sin((1 / 2) * theta) ** 8
                    + 84000 * np.sin((1 / 2) * theta) ** 6
                    - 71880 * np.sin((1 / 2) * theta) ** 4
                    - 1000 * np.sin((1 / 2) * theta) ** 2
                    + 6210 * np.sin(theta) ** 2
                    + 115640 * np.cos((1 / 2) * theta) ** 8
                    - 123200 * np.cos((1 / 2) * theta) ** 6
                    + 12168
                )
                * np.exp(5 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            )
            * np.exp(-5 * k**2 * sig_g**2)
            / (k**10 * sig_g**10)
        )

    return func


def K_gg_1_0(theta, phi, sig_g):
    def func(k):
        return -np.sqrt(2) * np.sqrt(np.pi) * np.exp(
            -1 / 2 * k**2 * sig_g**2
        ) * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) / (
            k**3 * sig_g**3
        ) + np.pi * np.erf(
            (1 / 2) * np.sqrt(2) * k * sig_g
        ) ** 2 / (
            k**4 * sig_g**4
        )

    return func


def K_gg_1_1(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_1_2(theta, phi, sig_g):
    def func(k):
        return (
            (15 / 28)
            * (
                (1 / 4)
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**5
                * sig_g**5
                * (
                    10 * np.cos(2 * theta)
                    + 3 * np.cos(2 * phi - theta)
                    + 3 * np.cos(2 * phi + theta)
                    - 8
                )
                * np.exp(3 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + (1 / 4)
                * np.pi
                * k**4
                * sig_g**4
                * (
                    5 * np.cos(2 * theta)
                    - 9 * np.cos(2 * phi - theta)
                    - 9 * np.cos(2 * phi + theta)
                    - 11
                )
                * np.exp((7 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + (1 / 2)
                * k**4
                * sig_g**4
                * (
                    30 * np.cos(2 * theta)
                    + 51 * np.cos(2 * phi - theta)
                    + 51 * np.cos(2 * phi + theta)
                    + 4
                )
                * np.exp((5 / 2) * k**2 * sig_g**2)
                + (5 / 4)
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**3
                * sig_g**3
                * (
                    6 * np.cos(2 * theta)
                    - 15 * np.cos(2 * phi - theta)
                    - 15 * np.cos(2 * phi + theta)
                    - 16
                )
                * np.exp(3 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + np.pi
                * k**2
                * sig_g**2
                * (
                    -15 * np.cos(2 * theta)
                    + 6 * np.cos(2 * phi - theta)
                    + 6 * np.cos(2 * phi + theta)
                    + 19
                )
                * np.exp((7 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + (45 / 2)
                * k**2
                * sig_g**2
                * (
                    3 * np.cos(2 * theta)
                    + 3 * np.cos(2 * phi - theta)
                    + 3 * np.cos(2 * phi + theta)
                    - 1
                )
                * np.exp((5 / 2) * k**2 * sig_g**2)
                + (45 / 2)
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k
                * sig_g
                * (
                    -3 * np.cos(2 * theta)
                    - 3 * np.cos(2 * phi - theta)
                    - 3 * np.cos(2 * phi + theta)
                    + 1
                )
                * np.exp(3 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + (45 / 4)
                * np.pi
                * (
                    3 * np.cos(2 * theta)
                    + 3 * np.cos(2 * phi - theta)
                    + 3 * np.cos(2 * phi + theta)
                    - 1
                )
                * np.exp((7 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            )
            * np.exp(-7 / 2 * k**2 * sig_g**2)
            / (k**8 * sig_g**8)
        )

    return func


def K_gg_1_3(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_1_4(theta, phi, sig_g):
    def func(k):
        return (
            (45 / 2342912)
            * (
                8
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**9
                * sig_g**9
                * (
                    -1607445 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 2913540 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 1771490 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 1443400 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 140875 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 1607445 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 3516240 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 1641115 * np.sin(phi) ** 4
                    + 680400 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 950880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 367640 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 2485240 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 281750 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 680400 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 2604000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 1617680 * np.sin(phi) ** 2
                    + 149688 * np.sin((1 / 2) * theta) ** 8
                    - 608160 * np.sin((1 / 2) * theta) ** 6
                    + 977928 * np.sin((1 / 2) * theta) ** 4
                    - 403768 * np.sin((1 / 2) * theta) ** 2
                    + 149478 * np.sin(theta) ** 2
                    - 802872 * np.cos((1 / 2) * theta) ** 8
                    + 880320 * np.cos((1 / 2) * theta) ** 6
                    - 138632
                )
                * np.exp(6 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 48
                * k**8
                * sig_g**8
                * (
                    893025 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 1482740 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 1793050 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 7000 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 148225 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 893025 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 2089360 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 646415 * np.sin(phi) ** 4
                    - 378000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 780640 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 1743560 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 60520 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 296450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 378000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 1815520 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 809040 * np.sin(phi) ** 2
                    - 83160 * np.sin((1 / 2) * theta) ** 8
                    + 182560 * np.sin((1 / 2) * theta) ** 6
                    + 605976 * np.sin((1 / 2) * theta) ** 4
                    - 671656 * np.sin((1 / 2) * theta) ** 2
                    + 163506 * np.sin(theta) ** 2
                    + 446040 * np.cos((1 / 2) * theta) ** 8
                    - 333760 * np.cos((1 / 2) * theta) ** 6
                    + 13416
                )
                * np.exp((11 / 2) * k**2 * sig_g**2)
                + np.pi
                * k**8
                * sig_g**8
                * (
                    4822335 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 9199260 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 11553430 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 1908760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 1055215 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 4822335 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 10090080 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 3576545 * np.sin(phi) ** 4
                    - 2041200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 2000880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 10261160 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6258760 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 2110430 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 2041200 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 6567120 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 2593120 * np.sin(phi) ** 2
                    - 449064 * np.sin((1 / 2) * theta) ** 8
                    + 2348640 * np.sin((1 / 2) * theta) ** 6
                    + 3373608 * np.sin((1 / 2) * theta) ** 4
                    - 6271288 * np.sin((1 / 2) * theta) ** 2
                    + 1217958 * np.sin(theta) ** 2
                    + 2408616 * np.cos((1 / 2) * theta) ** 8
                    - 3165120 * np.cos((1 / 2) * theta) ** 6
                    + 1143064
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 6
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**7
                * sig_g**7
                * (
                    -2679075 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 6129900 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 9463230 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 4105080 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 115885 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 2679075 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 4586400 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 2339645 * np.sin(phi) ** 4
                    + 1134000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 781200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 4189640 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 6233320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 231770 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 1134000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 882000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 242080 * np.sin(phi) ** 2
                    + 249480 * np.sin((1 / 2) * theta) ** 8
                    - 2469600 * np.sin((1 / 2) * theta) ** 6
                    + 1577464 * np.sin((1 / 2) * theta) ** 4
                    + 2165976 * np.sin((1 / 2) * theta) ** 2
                    + 62034 * np.sin(theta) ** 2
                    - 1338120 * np.cos((1 / 2) * theta) ** 8
                    + 2923200 * np.cos((1 / 2) * theta) ** 6
                    - 1486264
                )
                * np.exp(6 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 24
                * np.pi
                * k**6
                * sig_g**6
                * (
                    -8037225 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 15026340 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 7408730 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 8665720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 773465 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 8037225 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 17122560 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 7819735 * np.sin(phi) ** 4
                    + 3402000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 3902640 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 1838200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6673960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 1546930 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 3402000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 11775120 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 6926720 * np.sin(phi) ** 2
                    + 748440 * np.sin((1 / 2) * theta) ** 8
                    - 3564960 * np.sin((1 / 2) * theta) ** 6
                    - 456792 * np.sin((1 / 2) * theta) ** 4
                    + 4365512 * np.sin((1 / 2) * theta) ** 2
                    - 919002 * np.sin(theta) ** 2
                    - 4014360 * np.cos((1 / 2) * theta) ** 8
                    + 4925760 * np.cos((1 / 2) * theta) ** 6
                    - 1200680
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 42
                * k**6
                * sig_g**6
                * (
                    32276475 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 55941900 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 22926190 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 41626760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 690795 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 32276475 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 73164000 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 31813285 * np.sin(phi) ** 4
                    - 13662000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 23847600 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 1559480 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 54726040 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 1381590 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 13662000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 59235600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 35203040 * np.sin(phi) ** 2
                    - 3005640 * np.sin((1 / 2) * theta) ** 8
                    + 9285600 * np.sin((1 / 2) * theta) ** 6
                    - 4491832 * np.sin((1 / 2) * theta) ** 4
                    - 1713688 * np.sin((1 / 2) * theta) ** 2
                    + 810558 * np.sin(theta) ** 2
                    + 16121160 * np.cos((1 / 2) * theta) ** 8
                    - 14750400 * np.cos((1 / 2) * theta) ** 6
                    + 703352
                )
                * np.exp((11 / 2) * k**2 * sig_g**2)
                + 42
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**5
                * sig_g**5
                * (
                    3444525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 4823700 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 31866590 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 38755640 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 1459395 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 3444525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 8954400 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 8020915 * np.sin(phi) ** 4
                    - 1458000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 4674000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 38646920 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 50555560 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 2918790 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 1458000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 9433200 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 10844960 * np.sin(phi) ** 2
                    - 320760 * np.sin((1 / 2) * theta) ** 8
                    - 319200 * np.sin((1 / 2) * theta) ** 6
                    - 9848968 * np.sin((1 / 2) * theta) ** 4
                    + 12304088 * np.sin((1 / 2) * theta) ** 2
                    - 1670958 * np.sin(theta) ** 2
                    + 1720440 * np.cos((1 / 2) * theta) ** 8
                    - 264000 * np.cos((1 / 2) * theta) ** 6
                    - 2030392
                )
                * np.exp(6 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 378
                * np.pi
                * k**4
                * sig_g**4
                * (
                    6506325 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 11974900 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 1266930 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 11745720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 300965 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 6506325 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 14050400 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 7013195 * np.sin(phi) ** 4
                    - 2754000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 3510800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 4218760 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 12115880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 601930 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 2754000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 10046000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 6685280 * np.sin(phi) ** 2
                    - 605880 * np.sin((1 / 2) * theta) ** 8
                    + 2669600 * np.sin((1 / 2) * theta) ** 6
                    - 1351624 * np.sin((1 / 2) * theta) ** 4
                    - 1309416 * np.sin((1 / 2) * theta) ** 2
                    + 373506 * np.sin(theta) ** 2
                    + 3249720 * np.cos((1 / 2) * theta) ** 8
                    - 3771200 * np.cos((1 / 2) * theta) ** 6
                    + 642824
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 1260
                * k**4
                * sig_g**4
                * (
                    9823275 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 17322060 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 948430 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 18698120 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 50715 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 9823275 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 21971040 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 10626805 * np.sin(phi) ** 4
                    - 4158000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 6707760 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 7649720 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 22831000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 101430 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 4158000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 17224080 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 11327840 * np.sin(phi) ** 2
                    - 914760 * np.sin((1 / 2) * theta) ** 8
                    + 3164640 * np.sin((1 / 2) * theta) ** 6
                    - 3054520 * np.sin((1 / 2) * theta) ** 4
                    + 665960 * np.sin((1 / 2) * theta) ** 2
                    + 81630 * np.sin(theta) ** 2
                    + 4906440 * np.cos((1 / 2) * theta) ** 8
                    - 4827840 * np.cos((1 / 2) * theta) ** 6
                    + 269048
                )
                * np.exp((11 / 2) * k**2 * sig_g**2)
                + 8190
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**3
                * sig_g**3
                * (
                    893025 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 1660260 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 422870 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 2208920 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 41265 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 893025 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 1911840 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 1043455 * np.sin(phi) ** 4
                    - 378000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 450960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 1204520 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 2240200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 82530 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 378000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 1333680 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 983840 * np.sin(phi) ** 2
                    - 83160 * np.sin((1 / 2) * theta) ** 8
                    + 385440 * np.sin((1 / 2) * theta) ** 6
                    - 276520 * np.sin((1 / 2) * theta) ** 4
                    - 112840 * np.sin((1 / 2) * theta) ** 2
                    + 51930 * np.sin(theta) ** 2
                    + 446040 * np.cos((1 / 2) * theta) ** 8
                    - 536640 * np.cos((1 / 2) * theta) ** 6
                    + 84968
                )
                * np.exp(6 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 5040
                * np.pi
                * k**2
                * sig_g**2
                * (
                    -2679075 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 4863180 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 568610 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 5926760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 73395 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 2679075 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 5853120 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 3023965 * np.sin(phi) ** 4
                    + 1134000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 1571280 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 2913560 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6494200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 146790 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 1134000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 4320240 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 3014720 * np.sin(phi) ** 2
                    + 249480 * np.sin((1 / 2) * theta) ** 8
                    - 1021920 * np.sin((1 / 2) * theta) ** 6
                    + 831160 * np.sin((1 / 2) * theta) ** 4
                    + 100120 * np.sin((1 / 2) * theta) ** 2
                    - 94590 * np.sin(theta) ** 2
                    - 1338120 * np.cos((1 / 2) * theta) ** 8
                    + 1475520 * np.cos((1 / 2) * theta) ** 6
                    - 171704
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 255150
                * k**2
                * sig_g**2
                * (
                    231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 257915 * np.sin(phi) ** 4
                    - 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 262560 * np.sin(phi) ** 2
                    - 21560 * np.sin((1 / 2) * theta) ** 8
                    + 84000 * np.sin((1 / 2) * theta) ** 6
                    - 71880 * np.sin((1 / 2) * theta) ** 4
                    - 1000 * np.sin((1 / 2) * theta) ** 2
                    + 6210 * np.sin(theta) ** 2
                    + 115640 * np.cos((1 / 2) * theta) ** 8
                    - 123200 * np.cos((1 / 2) * theta) ** 6
                    + 12168
                )
                * np.exp((11 / 2) * k**2 * sig_g**2)
                + 255150
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k
                * sig_g
                * (
                    -231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 257915 * np.sin(phi) ** 4
                    + 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 262560 * np.sin(phi) ** 2
                    + 21560 * np.sin((1 / 2) * theta) ** 8
                    - 84000 * np.sin((1 / 2) * theta) ** 6
                    + 71880 * np.sin((1 / 2) * theta) ** 4
                    + 1000 * np.sin((1 / 2) * theta) ** 2
                    - 6210 * np.sin(theta) ** 2
                    - 115640 * np.cos((1 / 2) * theta) ** 8
                    + 123200 * np.cos((1 / 2) * theta) ** 6
                    - 12168
                )
                * np.exp(6 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 127575
                * np.pi
                * (
                    231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 257915 * np.sin(phi) ** 4
                    - 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 262560 * np.sin(phi) ** 2
                    - 21560 * np.sin((1 / 2) * theta) ** 8
                    + 84000 * np.sin((1 / 2) * theta) ** 6
                    - 71880 * np.sin((1 / 2) * theta) ** 4
                    - 1000 * np.sin((1 / 2) * theta) ** 2
                    + 6210 * np.sin(theta) ** 2
                    + 115640 * np.cos((1 / 2) * theta) ** 8
                    - 123200 * np.cos((1 / 2) * theta) ** 6
                    + 12168
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            )
            * np.exp(-13 / 2 * k**2 * sig_g**2)
            / (k**12 * sig_g**12)
        )

    return func


def K_gg_2_0(theta, phi, sig_g):
    def func(k):
        return (
            np.exp(-(k**2) * sig_g**2) / (k**4 * sig_g**4)
            - np.sqrt(2)
            * np.sqrt(np.pi)
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            / (k**5 * sig_g**5)
            + (1 / 2)
            * np.pi
            * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            / (k**6 * sig_g**6)
        )

    return func


def K_gg_2_1(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_2_2(theta, phi, sig_g):
    def func(k):
        return (
            (15 / 56)
            * (
                2
                * k**6
                * sig_g**6
                * (
                    5 * np.cos(2 * theta)
                    + 12 * np.cos(2 * phi - theta)
                    + 12 * np.cos(2 * phi + theta)
                    + 3
                )
                * np.exp((7 / 2) * k**2 * sig_g**2)
                + (1 / 2)
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**5
                * sig_g**5
                * (
                    10 * np.cos(2 * theta)
                    + 3 * np.cos(2 * phi - theta)
                    + 3 * np.cos(2 * phi + theta)
                    - 8
                )
                * np.exp(4 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + (1 / 4)
                * np.pi
                * k**4
                * sig_g**4
                * (
                    5 * np.cos(2 * theta)
                    - 9 * np.cos(2 * phi - theta)
                    - 9 * np.cos(2 * phi + theta)
                    - 11
                )
                * np.exp((9 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 3
                * k**4
                * sig_g**4
                * (
                    30 * np.cos(2 * theta)
                    + 51 * np.cos(2 * phi - theta)
                    + 51 * np.cos(2 * phi + theta)
                    + 4
                )
                * np.exp((7 / 2) * k**2 * sig_g**2)
                - 3
                / 2
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**3
                * sig_g**3
                * (
                    15 * np.cos(2 * theta)
                    + 57 * np.cos(2 * phi - theta)
                    + 57 * np.cos(2 * phi + theta)
                    + 23
                )
                * np.exp(4 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + (3 / 2)
                * np.pi
                * k**2
                * sig_g**2
                * (
                    -15 * np.cos(2 * theta)
                    + 6 * np.cos(2 * phi - theta)
                    + 6 * np.cos(2 * phi + theta)
                    + 19
                )
                * np.exp((9 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + (135 / 2)
                * k**2
                * sig_g**2
                * (
                    3 * np.cos(2 * theta)
                    + 3 * np.cos(2 * phi - theta)
                    + 3 * np.cos(2 * phi + theta)
                    - 1
                )
                * np.exp((7 / 2) * k**2 * sig_g**2)
                + (135 / 2)
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k
                * sig_g
                * (
                    -3 * np.cos(2 * theta)
                    - 3 * np.cos(2 * phi - theta)
                    - 3 * np.cos(2 * phi + theta)
                    + 1
                )
                * np.exp(4 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + (135 / 4)
                * np.pi
                * (
                    3 * np.cos(2 * theta)
                    + 3 * np.cos(2 * phi - theta)
                    + 3 * np.cos(2 * phi + theta)
                    - 1
                )
                * np.exp((9 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            )
            * np.exp(-9 / 2 * k**2 * sig_g**2)
            / (k**10 * sig_g**10)
        )

    return func


def K_gg_2_3(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gg_2_4(theta, phi, sig_g):
    def func(k):
        return (
            (45 / 4685824)
            * (
                128
                * k**10
                * sig_g**10
                * (
                    535815 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 920220 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 893830 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 177800 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 37975 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 535815 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 1223040 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 447545 * np.sin(phi) ** 4
                    - 226800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 411600 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 607880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 287960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 75950 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 226800 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 1006320 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 505600 * np.sin(phi) ** 2
                    - 49896 * np.sin((1 / 2) * theta) ** 8
                    + 144480 * np.sin((1 / 2) * theta) ** 6
                    + 105000 * np.sin((1 / 2) * theta) ** 4
                    - 201400 * np.sin((1 / 2) * theta) ** 2
                    + 42630 * np.sin(theta) ** 2
                    + 267624 * np.cos((1 / 2) * theta) ** 8
                    - 235200 * np.cos((1 / 2) * theta) ** 6
                    + 22360
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                + 16
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**9
                * sig_g**9
                * (
                    -1607445 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 2913540 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 1771490 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 1443400 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 140875 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 1607445 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 3516240 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 1641115 * np.sin(phi) ** 4
                    + 680400 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 950880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 367640 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 2485240 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 281750 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 680400 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 2604000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 1617680 * np.sin(phi) ** 2
                    + 149688 * np.sin((1 / 2) * theta) ** 8
                    - 608160 * np.sin((1 / 2) * theta) ** 6
                    + 977928 * np.sin((1 / 2) * theta) ** 4
                    - 403768 * np.sin((1 / 2) * theta) ** 2
                    + 149478 * np.sin(theta) ** 2
                    - 802872 * np.cos((1 / 2) * theta) ** 8
                    + 880320 * np.cos((1 / 2) * theta) ** 6
                    - 138632
                )
                * np.exp(7 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + np.pi
                * k**8
                * sig_g**8
                * (
                    4822335 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 9199260 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 11553430 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 1908760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 1055215 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 4822335 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 10090080 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 3576545 * np.sin(phi) ** 4
                    - 2041200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 2000880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 10261160 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6258760 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 2110430 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 2041200 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 6567120 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 2593120 * np.sin(phi) ** 2
                    - 449064 * np.sin((1 / 2) * theta) ** 8
                    + 2348640 * np.sin((1 / 2) * theta) ** 6
                    + 3373608 * np.sin((1 / 2) * theta) ** 4
                    - 6271288 * np.sin((1 / 2) * theta) ** 2
                    + 1217958 * np.sin(theta) ** 2
                    + 2408616 * np.cos((1 / 2) * theta) ** 8
                    - 3165120 * np.cos((1 / 2) * theta) ** 6
                    + 1143064
                )
                * np.exp((15 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 96
                * k**8
                * sig_g**8
                * (
                    15181425 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 26429620 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 12951610 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 17411240 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 481425 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 15181425 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 34296080 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 14658175 * np.sin(phi) ** 4
                    - 6426000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 10999520 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 1860040 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 22558360 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 962850 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 6426000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 27544160 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 16025040 * np.sin(phi) ** 2
                    - 1413720 * np.sin((1 / 2) * theta) ** 8
                    + 4501280 * np.sin((1 / 2) * theta) ** 6
                    - 1323112 * np.sin((1 / 2) * theta) ** 4
                    - 1890088 * np.sin((1 / 2) * theta) ** 2
                    + 560658 * np.sin(theta) ** 2
                    + 7582680 * np.cos((1 / 2) * theta) ** 8
                    - 7071680 * np.cos((1 / 2) * theta) ** 6
                    + 507624
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                + 6
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**7
                * sig_g**7
                * (
                    83051325 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 144570580 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 13837250 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 152265400 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 2573235 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 83051325 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 187634720 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 89913635 * np.sin(phi) ** 4
                    - 35154000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 60201680 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 67642120 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 204067880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 5146470 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 35154000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 150723440 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 98804000 * np.sin(phi) ** 2
                    - 7733880 * np.sin((1 / 2) * theta) ** 8
                    + 24607520 * np.sin((1 / 2) * theta) ** 6
                    - 35401352 * np.sin((1 / 2) * theta) ** 4
                    + 19244632 * np.sin((1 / 2) * theta) ** 2
                    - 2735502 * np.sin(theta) ** 2
                    + 41481720 * np.cos((1 / 2) * theta) ** 8
                    - 38669120 * np.cos((1 / 2) * theta) ** 6
                    + 540488
                )
                * np.exp(7 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 36
                * np.pi
                * k**6
                * sig_g**6
                * (
                    -8037225 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 15026340 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 7408730 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 8665720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 773465 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 8037225 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 17122560 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 7819735 * np.sin(phi) ** 4
                    + 3402000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 3902640 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 1838200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6673960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 1546930 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 3402000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 11775120 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 6926720 * np.sin(phi) ** 2
                    + 748440 * np.sin((1 / 2) * theta) ** 8
                    - 3564960 * np.sin((1 / 2) * theta) ** 6
                    - 456792 * np.sin((1 / 2) * theta) ** 4
                    + 4365512 * np.sin((1 / 2) * theta) ** 2
                    - 919002 * np.sin(theta) ** 2
                    - 4014360 * np.cos((1 / 2) * theta) ** 8
                    + 4925760 * np.cos((1 / 2) * theta) ** 6
                    - 1200680
                )
                * np.exp((15 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 126
                * k**6
                * sig_g**6
                * (
                    132890625 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 234028900 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 52586730 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 213194520 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 1496145 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 132890625 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 297533600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 138271455 * np.sin(phi) ** 4
                    - 56250000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 91312400 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 59964840 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 265083720 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 2992290 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 56250000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 233841200 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 147452320 * np.sin(phi) ** 2
                    - 12375000 * np.sin((1 / 2) * theta) ** 8
                    + 42461600 * np.sin((1 / 2) * theta) ** 6
                    - 32778792 * np.sin((1 / 2) * theta) ** 4
                    + 338232 * np.sin((1 / 2) * theta) ** 2
                    + 1993818 * np.sin(theta) ** 2
                    + 66375000 * np.cos((1 / 2) * theta) ** 8
                    - 64961600 * np.cos((1 / 2) * theta) ** 6
                    + 4614376
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                + 126
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**5
                * sig_g**5
                * (
                    15946875 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 29353100 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 43731730 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 75625480 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 330645 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 15946875 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 34434400 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 23579045 * np.sin(phi) ** 4
                    - 6750000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 8599600 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 61455160 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 82624280 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 661290 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 6750000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 24614800 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 23683680 * np.sin(phi) ** 2
                    - 1485000 * np.sin((1 / 2) * theta) ** 8
                    + 6546400 * np.sin((1 / 2) * theta) ** 6
                    - 13489208 * np.sin((1 / 2) * theta) ** 4
                    + 7873768 * np.sin((1 / 2) * theta) ** 2
                    - 274818 * np.sin(theta) ** 2
                    + 7965000 * np.cos((1 / 2) * theta) ** 8
                    - 9246400 * np.cos((1 / 2) * theta) ** 6
                    + 117624
                )
                * np.exp(7 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 18900
                * k**4
                * sig_g**4
                * (
                    5060475 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 9016140 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 62930 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 10183880 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 65835 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 5060475 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 11225760 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 5558245 * np.sin(phi) ** 4
                    - 2142000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 3283440 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 4492280 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 11939800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 131670 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 2142000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 8621520 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 5785760 * np.sin(phi) ** 2
                    - 471240 * np.sin((1 / 2) * theta) ** 8
                    + 1736160 * np.sin((1 / 2) * theta) ** 6
                    - 1572280 * np.sin((1 / 2) * theta) ** 4
                    + 155240 * np.sin((1 / 2) * theta) ** 2
                    + 90270 * np.sin(theta) ** 2
                    + 2527560 * np.cos((1 / 2) * theta) ** 8
                    - 2592960 * np.cos((1 / 2) * theta) ** 6
                    + 204152
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                + 126
                * np.pi
                * k**4
                * sig_g**4
                * (
                    47840625 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 87913700 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 2436490 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 93244760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 2240385 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 47840625 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 103448800 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 52362415 * np.sin(phi) ** 4
                    - 20250000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 26069200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 37554920 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 96436360 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 4480770 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 20250000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 74239600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 50280160 * np.sin(phi) ** 2
                    - 4455000 * np.sin((1 / 2) * theta) ** 8
                    + 19472800 * np.sin((1 / 2) * theta) ** 6
                    - 10553896 * np.sin((1 / 2) * theta) ** 4
                    - 8605384 * np.sin((1 / 2) * theta) ** 2
                    + 2771034 * np.sin(theta) ** 2
                    + 23895000 * np.cos((1 / 2) * theta) ** 8
                    - 27572800 * np.cos((1 / 2) * theta) ** 6
                    + 4419688
                )
                * np.exp((15 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 9450
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**3
                * sig_g**3
                * (
                    297675 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 710220 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 1074290 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 1669640 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 80955 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 297675 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 480480 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 489685 * np.sin(phi) ** 4
                    - 126000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 140880 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 1334840 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 1048600 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 161910 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 126000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 18960 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 243680 * np.sin(phi) ** 2
                    - 27720 * np.sin((1 / 2) * theta) ** 8
                    + 307680 * np.sin((1 / 2) * theta) ** 6
                    - 90040 * np.sin((1 / 2) * theta) ** 4
                    - 355480 * np.sin((1 / 2) * theta) ** 2
                    + 98910 * np.sin(theta) ** 2
                    + 148680 * np.cos((1 / 2) * theta) ** 8
                    - 358080 * np.cos((1 / 2) * theta) ** 6
                    + 139256
                )
                * np.exp(7 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 18900
                * np.pi
                * k**2
                * sig_g**2
                * (
                    -2679075 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 4863180 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 568610 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 5926760 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 73395 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 2679075 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 5853120 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 3023965 * np.sin(phi) ** 4
                    + 1134000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 1571280 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 2913560 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 6494200 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 146790 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 1134000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 4320240 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 3014720 * np.sin(phi) ** 2
                    + 249480 * np.sin((1 / 2) * theta) ** 8
                    - 1021920 * np.sin((1 / 2) * theta) ** 6
                    + 831160 * np.sin((1 / 2) * theta) ** 4
                    + 100120 * np.sin((1 / 2) * theta) ** 2
                    - 94590 * np.sin(theta) ** 2
                    - 1338120 * np.cos((1 / 2) * theta) ** 8
                    + 1475520 * np.cos((1 / 2) * theta) ** 6
                    - 171704
                )
                * np.exp((15 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
                + 1275750
                * k**2
                * sig_g**2
                * (
                    231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 257915 * np.sin(phi) ** 4
                    - 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 262560 * np.sin(phi) ** 2
                    - 21560 * np.sin((1 / 2) * theta) ** 8
                    + 84000 * np.sin((1 / 2) * theta) ** 6
                    - 71880 * np.sin((1 / 2) * theta) ** 4
                    - 1000 * np.sin((1 / 2) * theta) ** 2
                    + 6210 * np.sin(theta) ** 2
                    + 115640 * np.cos((1 / 2) * theta) ** 8
                    - 123200 * np.cos((1 / 2) * theta) ** 6
                    + 12168
                )
                * np.exp((13 / 2) * k**2 * sig_g**2)
                + 1275750
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k
                * sig_g
                * (
                    -231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    + 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    + 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    - 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    - 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    + 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    - 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    + 257915 * np.sin(phi) ** 4
                    + 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    - 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    - 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    + 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    + 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    - 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    + 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    - 262560 * np.sin(phi) ** 2
                    + 21560 * np.sin((1 / 2) * theta) ** 8
                    - 84000 * np.sin((1 / 2) * theta) ** 6
                    + 71880 * np.sin((1 / 2) * theta) ** 4
                    + 1000 * np.sin((1 / 2) * theta) ** 2
                    - 6210 * np.sin(theta) ** 2
                    - 115640 * np.cos((1 / 2) * theta) ** 8
                    + 123200 * np.cos((1 / 2) * theta) ** 6
                    - 12168
                )
                * np.exp(7 * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 637875
                * np.pi
                * (
                    231525 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 8
                    - 416500 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 6
                    - 26670 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 4
                    + 489720 * np.sin(phi) ** 4 * np.sin((1 / 2) * theta) ** 2
                    + 4725 * np.sin(phi) ** 4 * np.sin(theta) ** 2
                    - 231525 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 8
                    + 509600 * np.sin(phi) ** 4 * np.cos((1 / 2) * theta) ** 6
                    - 257915 * np.sin(phi) ** 4
                    - 98000 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 8
                    + 142800 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 6
                    + 229320 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 4
                    - 553960 * np.sin(phi) ** 2 * np.sin((1 / 2) * theta) ** 2
                    - 9450 * np.sin(phi) ** 2 * np.sin(theta) ** 2
                    + 98000 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 8
                    - 383600 * np.sin(phi) ** 2 * np.cos((1 / 2) * theta) ** 6
                    + 262560 * np.sin(phi) ** 2
                    - 21560 * np.sin((1 / 2) * theta) ** 8
                    + 84000 * np.sin((1 / 2) * theta) ** 6
                    - 71880 * np.sin((1 / 2) * theta) ** 4
                    - 1000 * np.sin((1 / 2) * theta) ** 2
                    + 6210 * np.sin(theta) ** 2
                    + 115640 * np.cos((1 / 2) * theta) ** 8
                    - 123200 * np.cos((1 / 2) * theta) ** 6
                    + 12168
                )
                * np.exp((15 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g) ** 2
            )
            * np.exp(-15 / 2 * k**2 * sig_g**2)
            / (k**14 * sig_g**14)
        )

    return func


def K_gv_0_0(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gv_0_1(theta, phi, sig_g):
    def func(k):
        return (
            (3 / 4)
            * (
                (1 / 2)
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (np.cos(phi - 3 / 2 * theta) - np.cos(phi + (1 / 2) * theta))
                * np.exp((1 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + k
                * sig_g
                * (3 * np.cos(phi - 3 / 2 * theta) + np.cos(phi + (1 / 2) * theta))
                - 1
                / 2
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * (3 * np.cos(phi - 3 / 2 * theta) + np.cos(phi + (1 / 2) * theta))
                * np.exp((1 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            )
            * np.exp(-1 / 2 * k**2 * sig_g**2)
            / (k**4 * sig_g**3)
        )

    return func


def K_gv_0_2(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_gv_0_3(theta, phi, sig_g):
    def func(k):
        return (
            (7 / 256)
            * (
                3
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**4
                * sig_g**4
                * (
                    -np.cos(phi - 3 / 2 * theta)
                    + np.cos(phi + (1 / 2) * theta)
                    - 5 * np.cos(3 * phi - 5 / 2 * theta)
                    + 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp(k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 2
                * k**3
                * sig_g**3
                * (
                    21 * np.cos(phi - 3 / 2 * theta)
                    + 27 * np.cos(phi + (1 / 2) * theta)
                    + 25 * np.cos(3 * phi - 5 / 2 * theta)
                    + 55 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp((1 / 2) * k**2 * sig_g**2)
                + 6
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (
                    9 * np.cos(phi - 3 / 2 * theta)
                    + 3 * np.cos(phi + (1 / 2) * theta)
                    + 25 * np.cos(3 * phi - 5 / 2 * theta)
                    - 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp(k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 30
                * k
                * sig_g
                * (
                    15 * np.cos(phi - 3 / 2 * theta)
                    + 9 * np.cos(phi + (1 / 2) * theta)
                    + 35 * np.cos(3 * phi - 5 / 2 * theta)
                    + 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp((1 / 2) * k**2 * sig_g**2)
                - 15
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * (
                    15 * np.cos(phi - 3 / 2 * theta)
                    + 9 * np.cos(phi + (1 / 2) * theta)
                    + 35 * np.cos(3 * phi - 5 / 2 * theta)
                    + 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp(k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
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
            (3 / 4)
            * (
                2
                * k**3
                * sig_g**3
                * (np.cos(phi - 3 / 2 * theta) + np.cos(phi + (1 / 2) * theta))
                * np.exp((1 / 2) * k**2 * sig_g**2)
                + (1 / 2)
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (np.cos(phi - 3 / 2 * theta) - np.cos(phi + (1 / 2) * theta))
                * np.exp(k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 3
                * k
                * sig_g
                * (3 * np.cos(phi - 3 / 2 * theta) + np.cos(phi + (1 / 2) * theta))
                * np.exp((1 / 2) * k**2 * sig_g**2)
                - 3
                / 2
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * (3 * np.cos(phi - 3 / 2 * theta) + np.cos(phi + (1 / 2) * theta))
                * np.exp(k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            )
            * np.exp(-(k**2) * sig_g**2)
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
            (7 / 256)
            * (
                16
                * k**5
                * sig_g**5
                * (
                    3 * np.cos(phi - 3 / 2 * theta)
                    + 3 * np.cos(phi + (1 / 2) * theta)
                    + 5 * np.cos(3 * phi - 5 / 2 * theta)
                    + 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp(k**2 * sig_g**2)
                + 3
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**4
                * sig_g**4
                * (
                    -np.cos(phi - 3 / 2 * theta)
                    + np.cos(phi + (1 / 2) * theta)
                    - 5 * np.cos(3 * phi - 5 / 2 * theta)
                    + 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp((3 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 2
                * k**3
                * sig_g**3
                * (
                    213 * np.cos(phi - 3 / 2 * theta)
                    + 171 * np.cos(phi + (1 / 2) * theta)
                    + 425 * np.cos(3 * phi - 5 / 2 * theta)
                    + 215 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp(k**2 * sig_g**2)
                + 18
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * k**2
                * sig_g**2
                * (
                    9 * np.cos(phi - 3 / 2 * theta)
                    + 3 * np.cos(phi + (1 / 2) * theta)
                    + 25 * np.cos(3 * phi - 5 / 2 * theta)
                    - 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp((3 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
                + 150
                * k
                * sig_g
                * (
                    15 * np.cos(phi - 3 / 2 * theta)
                    + 9 * np.cos(phi + (1 / 2) * theta)
                    + 35 * np.cos(3 * phi - 5 / 2 * theta)
                    + 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp(k**2 * sig_g**2)
                - 75
                * np.sqrt(2)
                * np.sqrt(np.pi)
                * (
                    15 * np.cos(phi - 3 / 2 * theta)
                    + 9 * np.cos(phi + (1 / 2) * theta)
                    + 35 * np.cos(3 * phi - 5 / 2 * theta)
                    + 5 * np.cos(3 * phi - 1 / 2 * theta)
                )
                * np.exp((3 / 2) * k**2 * sig_g**2)
                * np.erf((1 / 2) * np.sqrt(2) * k * sig_g)
            )
            * np.exp(-3 / 2 * k**2 * sig_g**2)
            / (k**8 * sig_g**7)
        )

    return func


def K_gv_1_4(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_vv_0_0(theta, phi, sig_g):
    def func(k):
        return (1 / 3) * np.cos(theta) / k**2

    return func


def K_vv_0_1(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_vv_0_2(theta, phi, sig_g):
    def func(k):
        return (1 / 6) * (3 * np.cos(2 * phi) + np.cos(theta)) / k**2

    return func


def K_vv_0_3(theta, phi, sig_g):
    def func(k):
        return 0

    return func


def K_vv_0_4(theta, phi, sig_g):
    def func(k):
        return 0

    return func
