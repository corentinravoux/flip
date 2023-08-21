import numpy as np


def K_gg_0_0(theta,phi,k,sig_g):
    def func(k):
        return((1/2)*np.sqrt(np.pi)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g))/(k**2*sig_g**2))
    return(func)

def K_gg_0_1(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gg_0_2(theta,phi,k,sig_g):
    def func(k):
        return(-5/32*(3*np.cos(2*phi) + 1)*(2*np.sqrt(np.pi)*k**2*sig_g**2*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)) + 6*k**2*sig_g**2 - 3*np.sqrt(np.pi)*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)))*np.exp(-k**2*sig_g**2)/(k**4*sig_g**4))
    return(func)

def K_gg_0_3(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gg_0_4(theta,phi,k,sig_g):
    def func(k):
        return((9/4096)*(20*np.cos(2*phi) + 35*np.cos(4*phi) + 9)*(12*np.sqrt(np.pi)*k**4*sig_g**4*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)) - 20*k**4*sig_g**4 - 60*np.sqrt(np.pi)*k**2*sig_g**2*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)) - 210*k**2*sig_g**2 + 105*np.sqrt(np.pi)*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)))*np.exp(-k**2*sig_g**2)/(k**6*sig_g**6))
    return(func)

def K_gg_1_0(theta,phi,k,sig_g):
    def func(k):
        return(-1/4*(2*k**2*sig_g**2 - np.sqrt(np.pi)*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)))*np.exp(-k**2*sig_g**2)/(k**4*sig_g**4))
    return(func)

def K_gg_1_1(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gg_1_2(theta,phi,k,sig_g):
    def func(k):
        return(-5/64*(3*np.cos(2*phi) + 1)*(8*k**4*sig_g**4 + 2*np.sqrt(np.pi)*k**2*sig_g**2*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)) + 18*k**2*sig_g**2 - 9*np.sqrt(np.pi)*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)))*np.exp(-k**2*sig_g**2)/(k**6*sig_g**6))
    return(func)

def K_gg_1_3(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gg_1_4(theta,phi,k,sig_g):
    def func(k):
        return(-9/8192*(20*np.cos(2*phi) + 35*np.cos(4*phi) + 9)*(64*k**6*sig_g**6 - 12*np.sqrt(np.pi)*k**4*sig_g**4*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)) + 340*k**4*sig_g**4 + 180*np.sqrt(np.pi)*k**2*sig_g**2*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)) + 1050*k**2*sig_g**2 - 525*np.sqrt(np.pi)*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)))*np.exp(-k**2*sig_g**2)/(k**8*sig_g**8))
    return(func)

def K_gg_2_0(theta,phi,k,sig_g):
    def func(k):
        return(-1/8*(4*k**4*sig_g**4 + 6*k**2*sig_g**2 - 3*np.sqrt(np.pi)*np.exp(k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf(abs(k)*abs(sig_g)))*np.exp(-k**2*sig_g**2)/(k**6*sig_g**6))
    return(func)

def K_gg_2_1(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gg_2_2(theta,phi,k,sig_g):
    def func(k):
        return(-5/128*(3*np.cos(2*phi) + 1)*(16*k**4*sig_g**4*abs(k)*abs(sig_g) + 6*np.sqrt(np.pi)*k**2*sig_g**2*np.exp(k**2*sig_g**2)*np.erf(abs(k)*abs(sig_g)) + 48*k**2*sig_g**2*abs(k)*abs(sig_g) - 45*np.sqrt(np.pi)*np.exp(k**2*sig_g**2)*np.erf(abs(k)*abs(sig_g)) + 90*abs(k)*abs(sig_g))*np.exp(-k**2*sig_g**2)/(k**6*sig_g**6*abs(k)*abs(sig_g)))
    return(func)

def K_gg_2_3(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gg_2_4(theta,phi,k,sig_g):
    def func(k):
        return(-9/16384*(20*np.cos(2*phi) + 35*np.cos(4*phi) + 9)*(128*k**6*sig_g**6*abs(k)*abs(sig_g) - 36*np.sqrt(np.pi)*k**4*sig_g**4*np.exp(k**2*sig_g**2)*np.erf(abs(k)*abs(sig_g)) + 832*k**4*sig_g**4*abs(k)*abs(sig_g) + 900*np.sqrt(np.pi)*k**2*sig_g**2*np.exp(k**2*sig_g**2)*np.erf(abs(k)*abs(sig_g)) + 3100*k**2*sig_g**2*abs(k)*abs(sig_g) - 3675*np.sqrt(np.pi)*np.exp(k**2*sig_g**2)*np.erf(abs(k)*abs(sig_g)) + 7350*abs(k)*abs(sig_g))*np.exp(-k**2*sig_g**2)/(k**8*sig_g**8*abs(k)*abs(sig_g)))
    return(func)

def K_gv_3_0(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gv_3_1(theta,phi,k,sig_g):
    def func(k):
        return(-3/2*(-np.sqrt(2)*np.sqrt(np.pi)*np.exp((1/2)*k**2*sig_g**2)*np.erf((1/2)*np.sqrt(2)*abs(k)*abs(sig_g)) + 2*abs(k)*abs(sig_g))*np.exp(-1/2*k**2*sig_g**2)*np.cos(phi)/(k**3*sig_g**2*abs(k)*abs(sig_g)))
    return(func)

def K_gv_3_2(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gv_3_3(theta,phi,k,sig_g):
    def func(k):
        return(-7/32*(3*np.cos(phi) + 5*np.cos(3*phi))*(3*np.sqrt(2)*np.sqrt(np.pi)*k**2*sig_g**2*np.exp((1/2)*k**2*sig_g**2)*np.erf((1/2)*np.sqrt(2)*abs(k)*abs(sig_g)) + 4*k**2*sig_g**2*abs(k)*abs(sig_g) - 15*np.sqrt(2)*np.sqrt(np.pi)*np.exp((1/2)*k**2*sig_g**2)*np.erf((1/2)*np.sqrt(2)*abs(k)*abs(sig_g)) + 30*abs(k)*abs(sig_g))*np.exp(-1/2*k**2*sig_g**2)/(k**5*sig_g**4*abs(k)*abs(sig_g)))
    return(func)

def K_gv_3_4(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gv_4_0(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gv_4_1(theta,phi,k,sig_g):
    def func(k):
        return(-3/2*(2*k**4*sig_g**4 + 6*k**2*sig_g**2 - 3*np.sqrt(2)*np.sqrt(np.pi)*np.exp((1/2)*k**2*sig_g**2)*abs(k)*abs(sig_g)*np.erf((1/2)*np.sqrt(2)*abs(k)*abs(sig_g)))*np.exp(-1/2*k**2*sig_g**2)*np.cos(phi)/(k**7*sig_g**6))
    return(func)

def K_gv_4_2(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_gv_4_3(theta,phi,k,sig_g):
    def func(k):
        return(-7/32*(3*np.cos(phi) + 5*np.cos(3*phi))*(4*k**4*sig_g**4*abs(k)*abs(sig_g) + 9*np.sqrt(2)*np.sqrt(np.pi)*k**2*sig_g**2*np.exp((1/2)*k**2*sig_g**2)*np.erf((1/2)*np.sqrt(2)*abs(k)*abs(sig_g)) + 32*k**2*sig_g**2*abs(k)*abs(sig_g) - 75*np.sqrt(2)*np.sqrt(np.pi)*np.exp((1/2)*k**2*sig_g**2)*np.erf((1/2)*np.sqrt(2)*abs(k)*abs(sig_g)) + 150*abs(k)*abs(sig_g))*np.exp(-1/2*k**2*sig_g**2)/(k**7*sig_g**6*abs(k)*abs(sig_g)))
    return(func)

def K_gv_4_4(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_vv_5_0(theta,phi,k,sig_g):
    def func(k):
        return((1/3)/k**2)
    return(func)

def K_vv_5_1(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_vv_5_2(theta,phi,k,sig_g):
    def func(k):
        return((1/6)*(3*np.cos(2*phi) + 1)/k**2)
    return(func)

def K_vv_5_3(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)

def K_vv_5_4(theta,phi,k,sig_g):
    def func(k):
        return(0)
    return(func)


