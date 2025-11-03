from get_spectrum_weight import w_spec
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
def compute_S0_Sx_Sy_Sz(galaxy_type, L, model):

    S0, Sx, Sy, Sz = (np.zeros_like(ENERGY_BIN_CENTERS) for _ in range(4))

    for Zs in ZSS:
        for galaxy in get_galaxy_set(galaxy_type):
            for EGMF in range(20):
                data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZs(Zs)]}/events_{galaxy}_EGMF{EGMF:02d}.txt")
                weights = w_L(galaxy, L) * w_sim(data[:,10]) * w_spec(data[:,10], Zs, model)
                
                S0 += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = weights)[0]
                Sx += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = -data[:,6] * weights)[0]
                Sy += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = -data[:,7] * weights)[0]
                Sz += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = -data[:,8] * weights)[0]

    return S0, Sx, Sy, Sz

# ----------------------------------------------------------------------------------------------------
def compute_dipole_amplitude(galaxy_type, L, model):

    S0, Sx, Sy, Sz = compute_S0_Sx_Sy_Sz(galaxy_type, L, model)
    np.savetxt(f"{RESULTS_DIR}/dipole_amplitude_full_sky_{model}_{galaxy_type}_{L}.dat", np.column_stack((ENERGY_BIN_CENTERS * 1e18, 3 * np.sqrt(Sx**2 + Sy**2 + Sz**2) / S0)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for model in ['CF2017', 'CF2023']:
        for galaxy_type in ['RG', 'RG+SBG', 'SBG']:
            for L in ['L11', 'Lradio', 'Lgamma']:
                compute_dipole_amplitude(galaxy_type, L, model)

# ----------------------------------------------------------------------------------------------------