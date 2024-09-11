def get_hubble_norm(z, cosmo):
    return 100 * cosmo.efunc(z)

def get_rcom_zobs(z, cosmo):
    return cosmo.comoving_distance(z).value * cosmo.h
