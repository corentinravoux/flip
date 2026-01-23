def compute_hubble_norm_zobs(z, cosmo):
    """Compute normalized Hubble parameter at redshift.

    Args:
        z (array-like): Redshift values.
        cosmo: Astropy cosmology object with `efunc`.

    Returns:
        ndarray: `100 * E(z)`.
    """
    return 100 * cosmo.efunc(z)


def compute_rcom_zobs(z, cosmo):
    """Compute comoving distance times little-h at redshift.

    Args:
        z (array-like): Redshift values.
        cosmo: Astropy cosmology object with `comoving_distance` and `h`.

    Returns:
        ndarray: `Dm(z) * h` in Mpc/h.
    """
    return cosmo.comoving_distance(z).value * cosmo.h
