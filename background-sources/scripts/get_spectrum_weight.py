from scipy.integrate import quad
import numpy as np 

ZSS = [1, 2, 7, 14, 26]

EeV_to_eV = 1e18
eV_to_erg = 1.60218e-12

Emin = 10**17.8 # eV

E0 = 1e18

# ----------------------------------------------------------------------------------------------------
def iZs(Zs):

    try:
        return ZSS.index(Zs)
    except ValueError:
        raise ValueError(f"Zs ({Zs}) not found in ZSS.")

# ----------------------------------------------------------------------------------------------------
def w_spec(Es, Zs, model):

    Gmm, Rcut = generation_rate_parameters(Zs, model)

    Es = Es * EeV_to_eV

    mask_low = Es <= Zs * Rcut
    mask_high = ~mask_low

    w_spec = np.zeros_like(Es)

    w_spec[mask_low] = (Es[mask_low] / E0)**-Gmm
    w_spec[mask_high] = (Es[mask_high] / E0)**-Gmm * np.exp(1 - Es[mask_high] / (Zs * Rcut))

    if model == 'CF2023':
        if Zs == 1:
            return w_spec / (quad(integrand_w_spec, Emin, 1e23, args = (Zs, model))[0] * eV_to_erg**2) # * L0
        else:
            return w_spec * [0.0, 0.245, 0.681, 0.049, 0.025][iZs(Zs)] / (quad(integrand_w_spec, Emin, 1e23, args = (Zs, model))[0] * eV_to_erg**2) # * L0

    elif model == 'CF2017':
        return w_spec * [0.064, 0.467, 0.375, 0.094, 0.0][iZs(Zs)]

# ----------------------------------------------------------------------------------------------------
def integrand_w_spec(Es, Zs, model): # arXiv:2211.02857

    Gmm, Rcut = generation_rate_parameters(Zs, model)

    if Es <= Zs * Rcut: 
        return Es * (Es / E0)**-Gmm
    elif Es > Zs * Rcut: 
        return Es * (Es / E0)**-Gmm * np.exp(1 - Es / (Zs * Rcut))

# ----------------------------------------------------------------------------------------------------
def generation_rate_parameters(Zs, model): 

    if model == 'CF2023': # arXiv:2211.02857
        if Zs == 1:
            # L0 = 6.54e44 # erg Mpc^-3 yr^-1
            Gmm = 3.34
            Rcut = 10**19.3 # V
            return Gmm, Rcut

        else:
            # L0 = 5e44 # erg Mpc^-3 yr^-1
            Gmm = -1.47
            Rcut = 10**18.19 # V
            return Gmm, Rcut
        
    elif model == 'CF2017': # arXiv:1612.07155
        Gmm = 1.22
        Rcut = 10**18.72 # V
        return Gmm, Rcut 

# ----------------------------------------------------------------------------------------------------