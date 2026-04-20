import jax
import jax.numpy as jnp


def get_unity_3dcoords(ra, dec):
    """"""
    return spherical_to_cartesian(jnp.ones_like(ra), ra, dec)


def compute_wavenumber_grid(nbins, lsize):
    wavenumber_grid = jnp.moveaxis(
        jnp.stack(
            jnp.meshgrid(
                jnp.fft.fftfreq(nbins, lsize / nbins),
                jnp.fft.fftfreq(nbins, lsize / nbins),
                jnp.fft.rfftfreq(nbins, lsize / nbins),
                indexing="ij",
            )
        )
        * 2
        * jnp.pi,
        0,
        -1,
    )
    wavenumber_norm_squared = jnp.sum(wavenumber_grid**2, axis=-1)[..., None]
    wavenumber_norm_squared = wavenumber_norm_squared.at[0, 0, 0].set(1)
    return (
        wavenumber_grid,
        wavenumber_grid / wavenumber_norm_squared,
        wavenumber_norm_squared[..., 0],
    )


def cartesian_to_spherical(x, y, z):
    """Converts cartesian [x,y,z] to spherical [r, theta, phi] coordinates
    (in degrees).

    Parameters
    ----------
    vec: array
        x, y, z

    Returns
    -------
    array
        [r, theta, phi]
    """
    r = jnp.sqrt(x**2 + y**2 + z**2)
    return jnp.stack(
        [r, jnp.arctan2(y, x) * 180 / jnp.pi, jnp.arcsin(z / r) * 180 / jnp.pi]
    )


def spherical_to_cartesian(r, ra, dec):
    """Converts dist-radec [r,ra,dec] to cartersian [x, y, z] coordinates
    (in degrees).

    Parameters
    ----------
    vec: array
        r, ra, dec

    Returns
    -------
    array
        [x, y, z]
    """
    theta = jnp.pi / 2 - dec * jnp.pi / 180
    phi = ra * jnp.pi / 180

    x = r * jnp.sin(theta) * jnp.cos(phi)
    y = r * jnp.sin(theta) * jnp.sin(phi)
    z = r * jnp.cos(theta)

    return jnp.stack([x, y, z]).T


class FourierBox(object):

    def __init__(self, nbins, lsize):
        """initialize the cube (box) that handle Fourier transformations.

        This automatically load the kgrid using `compute_kgrid`

        Parameters
        ---------
        nbins: int
            size of the box

        lsize: float
            physical size of the box

        Returns
        -------
        instance
        """
        self._nbins = nbins
        self._lsize = lsize
        self._load_kgrid_()

        self._xcube, self._ycube, self._zcube = jnp.meshgrid(
            jnp.arange(nbins), jnp.arange(nbins), jnp.arange(nbins)
        )
        xyz = (
            jnp.vstack(
                [self._xcube.flatten(), self._ycube.flatten(), self._zcube.flatten()]
            )
            - self.get_centroid()[:, None]
        )
        self._dist_mpch, self._ra, self._dec = cartesian_to_spherical(*xyz)
        self._dist_mpch *= self.bins_to_physical  # in distance

        self._dist_mpch = self._dist_mpch.reshape(self.shape)
        self._ra = self._ra.reshape(self.shape)
        self._dec = self._dec.reshape(self.shape)

    def _load_kgrid_(self):
        """load the kgrid (k, d2v, k2)"""
        self._k, self._d2v, self._k2 = compute_wavenumber_grid(self.nbins, self.lsize)

    def get_centroid(self, physical_unit=False):
        """ """
        centroid = jnp.asarray(self.centroid)
        if physical_unit:
            centroid = centroid * self.bins_to_physical

        return centroid

    def get_voxel_coordinates(
        self, ids=None, centroid=None, physical_unit=False, as_spherical=False
    ):
        """

        Parameters
        ----------
        ids: 1d-array
            voxel ids

        centroid: 3-array
            = ignored if as_spherical is False =
            center of for the spherical coordinated (x0, y0, z0).
            Careful, centroid must be in physical units if physical_unit=True

        physical_unit: bool
            should the coordinate be given in box bin unit (False)
            or in physical units (True) ; see self.bins_to_physical

        as_spherical: bool
            do you want the coordinate in cartesian (x,yz) unit (False)
            or in spherical (dist, ra, dec) units (True).

        Returns
        -------
        (N,3)-array
            with n the number of input ids

        """
        # convert into coordinates
        X, Y, Z = jnp.meshgrid(
            jnp.linspace(0, self.nbins - 1, self.nbins),
            jnp.linspace(0, self.nbins - 1, self.nbins),
            jnp.linspace(0, self.nbins - 1, self.nbins),
        )

        x_flat, y_flat, z_flat = (
            X.reshape(self.shape_flat),
            Y.reshape(self.shape_flat),
            Z.reshape(self.shape_flat),
        )
        # selected coordinates
        if ids is not None:
            xyz = jnp.stack([x_flat[ids], y_flat[ids], z_flat[ids]])
        else:
            xyz = jnp.stack([x_flat, y_flat, z_flat])

        if physical_unit:
            xyz *= self.bins_to_physical

        if as_spherical:
            xyz = self.xyz_to_distradec(
                xyz, centroid=centroid, physical_unit=physical_unit
            )

        return xyz

    def xyz_to_distradec(self, xyz, centroid=None, physical_unit=False):
        """converts the x, y, z coordinate system (voxel id)
        into spherical distance, ra, dec

        Parameters
        ----------
        xyz: (N,3)-array
            cube coordinate system (x,y,z)

        centroid: 3-array
            coordinate of the centroid within the box

        physical_unit: bool
            = ignored if centroid is given =
            are the input coordinates in physical or box units ?

        Returns
        -------
        (N,3) array
            dist, ra, dec
        """
        if centroid is None:
            centroid = self.get_centroid(physical_unit=physical_unit)

        dx, dy, dz = (xyz.T - centroid).T
        xyz = cartesian_to_spherical(dx, dy, dz)
        return xyz

    def get_voxels_in_direction(
        self, ra, dec, dist_range=None, physical_unit=False, unique=False
    ):
        """get all voxels in given direction

        Parameters
        ----------
        ra: float, array
            right ascension (deg)

        dec: float, array
            reclination (deg)

        dist_range: None, (float, float)
            maximum direction to be considered along the direction
            (see phyiscal unit for unit)

        physical_unit: bool
            are the distance given in physical units

        Returns
        -------
        list, array
            Coordinates (i,j,k) of the pixels for the given ra, dec
            format:
            - uniques=False: (ntargets, 3, nspaxels)
            - uniques=True: list of (3, nspaxels_i) each targets has its own nspaxels_i.

        """
        ra = jnp.atleast_1d(ra)
        dec = jnp.atleast_1d(dec)

        ntrial = self.nbins * 10
        xyz = spherical_to_cartesian(
            jnp.linspace(*dist_range, ntrial), ra[:, None], dec[:, None]
        )
        if physical_unit:
            xyz /= self.lsize / self.nbins  # xyz now in ijk centered on (0,0,0)

        xyz += self.get_centroid(physical_unit=False)  # ijk centered on centroid
        volexin = jnp.asarray(xyz, dtype="int32")  # (ntrial, ntargets, 3)
        volexin = jnp.moveaxis(volexin, 0, -1)  # (ntargets, 3, ntrial)
        if unique:
            volexin = [jnp.unique(v, axis=-1) for v in volexin]  # loops over targets
        return volexin

    def draw_voxelid(self, size, seed, density=None):
        """randomly draw voxel-ids

        This uses jax.random.choice.

        Parameters
        ----------
        size: int
            number of coordinates to draw.

        seed: PRNGKey
            a PRNG key used as the random key.
            if unsure, do: seed = jax.random.PRNGKey(0)

        density: 3d-array
           3d-pdf of the box elements. If None, all assumed to be equal.
           must be the same shape as self.shape

        Returns
        -------
        N-array
            voxel id per target (N=size)

        See also
        --------
        draw_coordinates: randomly draw (x,y,z) box coordinates.
        """
        # Logic:
        # work on flatten cube (faster and needed for jax or np random.choice.
        # so, build voxel indexes (index), draw from it, and convert that into coord.

        # this will forces density to be of the good size
        if density is not None:
            density_flat = density.reshape(self.shape_flat)
            density_flat_normed = density_flat / density_flat.sum()
        else:
            density_flat_normed = None

        # sampled
        if isinstance(size, int):
            size = (size,)  # jax format

        sampled_voxels = jax.random.choice(
            seed, self.voxel_id, shape=size, p=density_flat_normed
        )
        return sampled_voxels

    def draw_coordinates(
        self, size, seed, density=None, as_spherical=False, physical_unit=False
    ):
        """randomly draw (x,y,z) box coordinates.

        This uses jax.random.choice through draw_voxelid()

        Parameters
        ----------
        size: int
            number of coordinates to draw.

        seed: PRNGKey
            a PRNG key used as the random key.
            if unsure, do: seed = jax.random.PRNGKey(0)

        density: 3d-array
           3d-pdf of the box elements. If None, all assumed to be equal.
           must be the same shape as self.shape

        Returns
        -------
        array (n,3)
            (x,y,z) size times

        See also
        --------
        draw_voxelid: randomly draw voxel-ids
        """
        # Logic:
        # work on flatten cube (faster and needed for jax or np random.choice.
        # so, build voxel indexes (index), draw from it, and convert that into coord.

        sampled_voxels = self.draw_voxelid(size, seed, density=density)
        return self.get_voxel_coordinates(
            sampled_voxels, as_spherical=as_spherical, physical_unit=physical_unit
        )

    # =============== #
    #   Properties    #
    # =============== #
    @property
    def nbins(self):
        """size of the box"""
        return self._nbins

    @property
    def lsize(self):
        """physical size of the box"""
        return self._lsize

    @property
    def kshape(self):
        """shape of the k-modes"""
        return self.k.shape

    @property
    def k(self):
        """indice of the wavenumber"""
        if not hasattr(self, "_k") or self._k is None:
            self._load_kgrid_()

        return self._k

    @property
    def d2v(self):
        """delta to velocity"""
        if not hasattr(self, "_d2v") or self._d2v is None:
            self._load_kgrid_()

        return self._d2v

    @property
    def k2(self):
        """norm of k"""
        if not hasattr(self, "_k2") or self._k2 is None:
            self._load_kgrid_()

        return self._k2

    @property
    def r_1d(self):
        """1d radius in phystical units"""
        return jnp.linspace(0, self.lsize, self.nbins) - self.lsize / 2

    @property
    def shape(self):
        """shape of the box (nbins, nbins, nbins)"""
        return (self.nbins, self.nbins, self.nbins)

    @property
    def shape_flat(self):
        """shape of the box (nbins, nbins, nbins)"""
        return jnp.prod(jnp.asarray(self.shape))

    @property
    def centroid(self):
        """ """
        return (self.nbins / 2, self.nbins / 2, self.nbins / 2)

    @property
    def bins_to_physical(self):
        """physical size of a bins (lsize/nbins)"""
        return self.lsize / self.nbins

    @property
    def voxel_id(self):
        """linear 1d array of the voxel"""

        return jnp.arange(self.shape_flat)

    @property
    def voxel_vertices(self):
        """get the verticies of a voxel (cube*bins_to_physical)"""
        return (
            jnp.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 1],
                ],
            )
            * self.bins_to_physical
        )
