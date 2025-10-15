from scipy.integrate import quad
import numpy as np 

GALAXIES_DIR = "../../galaxies"
RESULTS_DIR = "../results"
SIMULATIONS_DIR = "../../simulations/local-sources/events"

ENERGY_EDGES = np.logspace(0, 4, num = 80)
ENERGY_BIN_CENTERS = np.logspace(4./(2.*(len(ENERGY_EDGES) - 1)), 4. - 4./(2.*(len(ENERGY_EDGES) - 1)), num = len(ENERGY_EDGES) - 1)
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
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
def w_L(galaxy, L):
    
    data = np.genfromtxt(f"{GALAXIES_DIR}/galaxies.dat", dtype = None, encoding = 'utf-8')

    if L == 'L11':
        iL = 4 
    elif L == 'Lradio':
        iL = 5 
    elif L == 'Lgamma':
        iL = 6

    for isrc in range(len(data)):
        if galaxy == data[isrc][0]:
            return data[isrc][iL]

# ----------------------------------------------------------------------------------------------------
def w_sim(Es):

    return Es * EeV_to_eV / E0

# ----------------------------------------------------------------------------------------------------
def w_spec(Es, Zs):

    L0, Gmm, Rcut = generation_rate_parameters(Zs)

    Es = Es * EeV_to_eV

    mask_low = Es <= Zs * Rcut
    mask_high = ~mask_low

    w_spec = np.zeros_like(Es)

    w_spec[mask_low] = (Es[mask_low] / E0)**-Gmm
    w_spec[mask_high] = (Es[mask_high] / E0)**-Gmm * np.exp(1 - Es[mask_high] / (Zs * Rcut))

    if Zs == 1:
        return w_spec * L0 / (quad(integrand_w_spec, Emin, 1e23, args = (Zs))[0] * eV_to_erg**2)
    else:
        return w_spec * [0.0, 0.245, 0.681, 0.049, 0.025][iZs(Zs)] * L0 / (quad(integrand_w_spec, Emin, 1e23, args = (Zs))[0] * eV_to_erg**2)

# ----------------------------------------------------------------------------------------------------
def integrand_w_spec(Es, Zs):

    _, Gmm, Rcut = generation_rate_parameters(Zs)

    if Es <= Zs * Rcut: 
        return Es * (Es / E0)**-Gmm
    elif Es > Zs * Rcut: 
        return Es * (Es / E0)**-Gmm * np.exp(1 - Es / (Zs * Rcut))

# ----------------------------------------------------------------------------------------------------
def generation_rate_parameters(Zs): # arXiv:2211.02857

    if Zs == 1:
        L0 = 6.54e44 # erg Mpc^-3 yr^-1
        Gmm = 3.34
        Rcut = 10**19.3 # V
        return L0, Gmm, Rcut

    else:
        L0 = 5e44 # erg Mpc^-3 yr^-1
        Gmm = -1.47
        Rcut = 10**18.19 # V
        return L0, Gmm, Rcut
    
# ----------------------------------------------------------------------------------------------------
def get_galaxy_set(galaxy_type):

    if galaxy_type == 'RG':
        return ['CenA', 'ForA', 'VirA']
    
    elif galaxy_type == 'RG+SBG':
        return ['CenA', 'ForA', 'VirA', \
            'NGC253', 'M82', 'NGC4945', 'M83', 'IC342', 'NGC6946', 'NGC2903', 'NGC5055', 'NGC3628', 'NGC3627', \
            'NGC4631', 'NGC891', 'NGC3556', 'NGC660', 'NGC2146', 'NGC3079', 'NGC1068', 'NGC1365']
    
    elif galaxy_type == 'SBG':
        return ['NGC253', 'M82', 'NGC4945', 'M83', 'IC342', 'NGC6946', 'NGC2903', 'NGC5055', 'NGC3628', 'NGC3627', \
            'NGC4631', 'NGC891', 'NGC3556', 'NGC660', 'NGC2146', 'NGC3079', 'NGC1068', 'NGC1365']
    
# ----------------------------------------------------------------------------------------------------   
def compute_spectrum(galaxy_type, L):

    spec = np.zeros_like(ENERGY_BIN_CENTERS)
    
    for Zs in ZSS:
        for galaxy in get_galaxy_set(galaxy_type):
            for EGMF in range(20):
                data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZs(Zs)]}/events_{galaxy}_EGMF{EGMF:02d}.txt")
                weights = w_L(galaxy, L) * w_sim(data[:,10]) * w_spec(data[:,10], Zs)
                spec += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = weights)[0]

    return spec 

# ----------------------------------------------------------------------------------------------------
def write_spectrum(galaxy_type, L):

    spec = compute_spectrum(galaxy_type, L)  
    np.savetxt(f"{RESULTS_DIR}/spec_{galaxy_type}_{L}.dat", np.column_stack((ENERGY_BIN_CENTERS * 1e18, spec / (ENERGY_BIN_CENTERS * 1e18))), fmt = "%.15e")
  
# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for galaxy_type in ['RG', 'RG+SBG', 'SBG']:
        for L in ['L11', 'Lradio', 'Lgamma']:
             write_spectrum(galaxy_type, L)

# ----------------------------------------------------------------------------------------------------