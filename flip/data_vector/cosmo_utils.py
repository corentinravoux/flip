def compute_hubble_norm_zobs(z, cosmo):
    return 100 * cosmo.efunc(z)

def compute_rcom_zobs(z, cosmo):
    return cosmo.comoving_distance(z).value * cosmo.h
