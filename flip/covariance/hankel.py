### This code is adapted from cosmoprimo (https://github.com/cosmodesi/cosmoprimo)
### Special thanks to the GOAT Arnaud de Mattia.

import numpy as np
from scipy.special import loggamma


class BaseKernel(object):
    """Base kernel."""

    def __call__(self, z):
        return self.eval(z)

    def __eq__(self, other):
        return other.__class__ == self.__class__


class BaseBesselKernel(BaseKernel):
    """Base Bessel kernel."""

    def __init__(self, nu):
        self.nu = nu

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.nu == self.nu


class SphericalBesselJKernel(BaseBesselKernel):
    """(Mellin transform of) spherical Bessel kernel."""

    def eval(self, z):
        return np.exp(
            np.log(2) * (z - 1.5)
            + loggamma(0.5 * (self.nu + z))
            - loggamma(0.5 * (3 + self.nu - z))
        )


def pad(array, pad_width, axis=-1, extrap=0):
    """
    Pad array along ``axis``.

    Parameters
    ----------
    array : array_like
        Input array to be padded.

    pad_width : int, tuple of ints
        Number of points to be added on both sides of the array.
        Pass a tuple to differentiate between left and right sides.

    axis : int, default=-1
        Axis along which padding is to be applied.

    extrap : string, float, default=0
        If 'log', performs a log-log extrapolation.
        If 'edge', pad ``array`` with its edge values.
        Else, pad ``array`` with the provided value.
        Pass a tuple to differentiate between left and right sides.

    Returns
    -------
    array : array
        Padded array.
    """
    array = np.asarray(array)

    try:
        pad_width_left, pad_width_right = pad_width
    except (TypeError, ValueError):
        pad_width_left = pad_width_right = pad_width

    try:
        extrap_left, extrap_right = extrap
    except (TypeError, ValueError):
        extrap_left = extrap_right = extrap

    axis = axis % array.ndim
    to_axis = [1] * array.ndim
    to_axis[axis] = -1

    if extrap_left == "edge":
        end = np.take(array, [0], axis=axis)
        pad_left = np.repeat(end, pad_width_left, axis=axis)
    elif extrap_left == "log":
        end = np.take(array, [0], axis=axis)
        ratio = np.take(array, [1], axis=axis) / end
        exp = np.arange(-pad_width_left, 0).reshape(to_axis)
        pad_left = end * ratio**exp
    else:
        pad_left = np.full(
            array.shape[:axis] + (pad_width_left,) + array.shape[axis + 1 :],
            extrap_left,
        )

    if extrap_right == "edge":
        end = np.take(array, [-1], axis=axis)
        pad_right = np.repeat(end, pad_width_right, axis=axis)
    elif extrap_right == "log":
        end = np.take(array, [-1], axis=axis)
        ratio = np.take(array, [-2], axis=axis) / end
        exp = np.arange(1, pad_width_right + 1).reshape(to_axis)
        pad_right = end / ratio**exp
    else:
        pad_right = np.full(
            array.shape[:axis] + (pad_width_right,) + array.shape[axis + 1 :],
            extrap_right,
        )

    return np.concatenate([pad_left, array, pad_right], axis=axis)


class FFTlog(object):
    r"""
    Implementation of the FFTlog algorithm presented in https://jila.colorado.edu/~ajsh/FFTLog/, which computes the generic integral:

    .. math::

        G(y) = \int_{0}^{\infty} x dx F(x) K(xy)

    with :math:`F(x)` input function, :math:`K(xy)` a kernel.

    This transform is (mathematically) invariant under a power law transformation:

    .. math::

        G_{q}(y) = \int_{0}^{\infty} x dx F_{q}(x) K_{q}(xy)

    where :math:`F_{q}(x) = G(x)x^{-q}`, :math:`K_{q}(t) = K(t)t^{q}` and :math:`G_{q}(y) = G(y)y^{q}`.
    """

    def __init__(
        self,
        x,
        kernel,
        q=0,
        minfolds=2,
        lowring=True,
        xy=1,
        check_level=0,
    ):
        r"""
        Initialize :class:`FFTlog`, which can perform several transforms at once.

        Parameters
        ----------
        x : array_like
            Input log-spaced coordinates. Must be strictly increasing.
            If 1D, is broadcast to the number of provided kernels.

        kernel : callable, list of callables
            Mellin transform of the kernel:
            .. math:: U_{K}(z) = \int_{0}^{\infty} t^{z-1} K(t) dt
            If a list of kernels is provided, will perform all transforms at once.

        q : float, list of floats, default=0
            Power-law tilt(s) to regularise integration.

        minfolds : int, default=2
            The c is chosen with minimum :math:`n` chosen such that ``2**n > minfolds * x.size``.

        lowring : bool, default=True
            If ``True`` set output coordinates according to the low-ringing condition, otherwise set it with ``xy``.

        xy : float, list of floats, default=1
            Enforce the reciprocal product (i.e. ``x[0] * y[-1]``) of the input ``x`` and output ``y`` coordinates.

        check_level : int, default=0
            If non-zero run sanity checks on input.

        Note
        ----
        Kernel definition is different from that of https://jila.colorado.edu/~ajsh/FFTLog/, which uses (eq. 10):

        .. math:: U_{K}(z) = \int_{0}^{\infty} t^{z} K(t) dt

        Therefore, one should use :math:`q = 1` for Bessel functions to match :math:`q = 0` in  https://jila.colorado.edu/~ajsh/FFTLog/.
        """
        self.inparallel = isinstance(kernel, (tuple, list))
        if not self.inparallel:
            kernel = [kernel]
        self.kernel = list(kernel)
        if np.ndim(q) == 0:
            q = [q] * self.nparallel
        self.q = list(q)
        self.x = np.asarray(x)
        if not self.inparallel:
            self.x = self.x[None, :]
        elif self.x.ndim == 1:
            self.x = np.tile(self.x[None, :], (self.nparallel, 1))
        if np.ndim(xy) == 0:
            xy = [xy] * self.nparallel
        self.xy = list(xy)
        self.check_level = check_level
        if self.check_level:
            if len(self.x) != self.nparallel:
                raise ValueError("x and kernel must of same length")
            if len(self.q) != self.nparallel:
                raise ValueError("q and kernel must be lists of same length")
            if len(self.xy) != self.nparallel:
                raise ValueError("xy and kernel must be lists of same length")
        self.minfolds = minfolds
        self.lowring = lowring
        self.setup()

    @property
    def nparallel(self):
        """Number of transforms performed in parallel."""
        return len(self.kernel)

    def setup(self):
        """Set up u funtions."""
        self.size = self.x.shape[-1]
        self.delta = np.log(self.x[:, -1] / self.x[:, 0]) / (self.size - 1)

        nfolds = (self.size * self.minfolds - 1).bit_length()
        self.padded_size = 2**nfolds
        npad = self.padded_size - self.size
        self.padded_size_in_left, self.padded_size_in_right = (
            npad // 2,
            npad - npad // 2,
        )
        self.padded_size_out_left, self.padded_size_out_right = (
            npad - npad // 2,
            npad // 2,
        )

        if self.check_level:
            if not np.allclose(
                np.log(self.x[:, 1:] / self.x[:, :-1]), self.delta, rtol=1e-3
            ):
                raise ValueError("Input x must be log-spaced")
            if self.padded_size < self.size:
                raise ValueError("Convolution size must be larger than input x size")

        if self.lowring:
            self.lnxy = np.array(
                [
                    delta / np.pi * np.angle(kernel(q + 1j * np.pi / delta))
                    for kernel, delta, q in zip(self.kernel, self.delta, self.q)
                ]
            )
        else:
            self.lnxy = np.log(self.xy) + self.delta

        self.y = np.exp(self.lnxy - self.delta)[:, None] / self.x[:, ::-1]

        m = np.arange(0, self.padded_size // 2 + 1)
        self.padded_u, self.padded_prefactor, self.padded_postfactor = [], [], []
        self.padded_x = pad(
            self.x,
            (self.padded_size_in_left, self.padded_size_in_right),
            axis=-1,
            extrap="log",
        )
        self.padded_y = pad(
            self.y,
            (self.padded_size_out_left, self.padded_size_out_right),
            axis=-1,
            extrap="log",
        )
        prev_kernel, prev_q, prev_delta, prev_u = None, None, None, None
        for kernel, padded_x, padded_y, lnxy, delta, q in zip(
            self.kernel, self.padded_x, self.padded_y, self.lnxy, self.delta, self.q
        ):
            self.padded_prefactor.append(padded_x ** (-q))
            self.padded_postfactor.append(padded_y ** (-q))
            if kernel is prev_kernel and q == prev_q and delta == prev_delta:
                u = prev_u
            else:
                u = prev_u = kernel(q + 2j * np.pi / self.padded_size / delta * m)
            self.padded_u.append(
                u * np.exp(-2j * np.pi * lnxy / self.padded_size / delta * m)
            )
            prev_kernel, prev_q, prev_delta = kernel, q, delta
        self.padded_u = np.array(self.padded_u)
        self.padded_prefactor = np.array(self.padded_prefactor)
        self.padded_postfactor = np.array(self.padded_postfactor)

    def __call__(self, fun, extrap=0, keep_padding=False):
        """
        Perform the transforms.

        Parameters
        ----------
        fun : array_like
            Function to be transformed.
            Last dimensions should match (:attr:`nparallel`,len(x)) where ``len(x)`` is the size of the input x-coordinates.
            (if :attr:`nparallel` is 1, the only requirement is the last dimension to be (len(x))).

        extrap : float, string, default=0
            How to extrapolate function outside of  ``x`` range to fit the integration range.
            If 'log', performs a log-log extrapolation.
            If 'edge', pad ``fun`` with its edge values.
            Else, pad ``fun`` with the provided value.
            Pass a tuple to differentiate between left and right sides.

        keep_padding : bool, default=False
            Whether to return function padded to the number of points in the integral.
            By default, crop it to its original size.

        Returns
        -------
        y : array
            Array of new coordinates.

        fftloged : array
            Transformed function.
        """
        fun = np.asarray(fun)
        padded_fun = pad(
            fun,
            (self.padded_size_in_left, self.padded_size_in_right),
            axis=-1,
            extrap=extrap,
        )
        fftloged = (
            np.fft.irfft(
                (
                    np.fft.rfft(padded_fun * self.padded_prefactor) * self.padded_u
                ).conj(),
                n=self.size,
                axis=-1,
            )
            * self.padded_postfactor
        )

        if not keep_padding:
            y = self.y
            fftloged = fftloged[
                ..., self.padded_size_out_left : self.padded_size_out_left + self.size
            ]
        else:
            y = self.padded_y
        if not self.inparallel:
            y = y[0]
            fftloged = np.reshape(
                fftloged,
                fun.shape if not keep_padding else fun.shape[:-1] + (self.padded_size,),
            )
        return y, fftloged


class PowerToCorrelation(FFTlog):
    r"""
    Power spectrum to correlation function transform, defined as:

    .. math::
        \xi_{\ell}(s) = \frac{(-i)^{\ell}}{2 \pi^{2}} \int dk k^{2} P_{\ell}(k) j_{\ell}(ks)

    """

    def __init__(self, k, ell=0, q=0, complex=False, **kwargs):
        """
        Initialize power to correlation transform.

        Parameters
        ----------
        k : array_like
            Input log-spaced wavenumbers.
            If 1D, is broadcast to the number of provided ``ell``.

        ell : int, list of int, default=0
            Poles. If a list is provided, will perform all transforms at once.

        q : float, list of floats, default=0
            Power-law tilt(s) to regularise integration.

        complex : bool, default=False
            ``False`` assumes the imaginary part of odd power spectrum poles is provided.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        if np.ndim(ell) == 0:
            kernel = SphericalBesselJKernel(ell)
        else:
            kernel = [SphericalBesselJKernel(ell_) for ell_ in ell]
        FFTlog.__init__(self, k, kernel, q=1.5 + q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 / (2 * np.pi) ** 1.5
        # Convention is (-i)^ell/(2 pi^2)
        ell = np.atleast_1d(ell)
        if complex:
            phase = (-1j) ** ell
        else:
            # Prefactor is (-i)^ell, but we take in the imaginary part of odd power spectra, hence:
            # (-i)^ell = (-1)^(ell/2) if ell is even
            # (-i)^ell i = (-1)^(ell//2) if ell is odd
            phase = (-1) ** (ell // 2)
        # Not in-place as phase (and hence padded_postfactor) may be complex instead of float
        self.padded_postfactor = self.padded_postfactor * phase[:, None]


class CorrelationToPower(FFTlog):
    r"""
    Correlation function to power spectrum transform, defined as:

    .. math::
        P_{\ell}(k) = 4 \pi i^{\ell} \int ds s^{2} \xi_{\ell}(s) j_{\ell}(ks)

    """

    def __init__(self, s, ell=0, q=0, complex=False, **kwargs):
        """
        Initialize power to correlation transform.

        Parameters
        ----------
        s : array_like
            Input log-spaced separations.
            If 1D, is broadcast to the number of provided ``ell``.

        ell : int, list of int, default=0
            Poles. If a list is provided, will perform all transforms at once.

        q : float, list of floats, default=0
            Power-law tilt(s) to regularise integration.

        complex : bool, default=False
            ``False`` returns the real part of even poles, and the imaginary part of odd poles.

        kwargs : dict
            Arguments for :class:`FFTlog`.
        """
        if np.ndim(ell) == 0:
            kernel = SphericalBesselJKernel(ell)
        else:
            kernel = [SphericalBesselJKernel(ell_) for ell_ in ell]
        FFTlog.__init__(self, s, kernel, q=1.5 + q, **kwargs)
        self.padded_prefactor *= self.padded_x**3 * (2 * np.pi) ** 1.5
        # Convention is 4 \pi i^ell, and we return imaginary part of odd poles
        ell = np.atleast_1d(ell)
        if complex:
            phase = (-1j) ** ell
        else:
            # We return imaginary part of odd poles
            phase = (-1) ** (ell // 2)
        self.padded_postfactor = self.padded_postfactor * phase[:, None]
