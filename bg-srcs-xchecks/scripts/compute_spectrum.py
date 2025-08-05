from scipy.integrate import quad
from scipy.special import erf
from scipy.special import k1
import numpy as np 

# with open("base_dir.txt", "r") as f:
#     BASE_DIR = f.read().strip()
RESULTS_DIR = "../results"
SIMULATIONS_DIR = "../../simulations/background-sources/results/binning"

CTSS = np.logspace(0, 3.5, num = 71)
ESS = np.delete(np.logspace(0, 4, num = 81), 0)
ES = np.logspace(4./(2.*(len(ESS) - 1)), 4. - 4./(2.*(len(ESS) - 1)), num = len(ESS) - 1)
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZSS = [1, 2, 7, 14, 26]

Gmm = 2

lbd_coh = 1 # Mpc

# ----------------------------------------------------------------------------------------------------
def iZs(Zs):

    try:
        return ZSS.index(Zs)
    except ValueError:
        raise ValueError(f"Zs ({Zs}) not found in ZSS.")

# ----------------------------------------------------------------------------------------------------
def larmor_radius(Es, Zs, B): # E in EeV and B in nG 

    return 1.081 / Zs * Es / B # Mpc

# ----------------------------------------------------------------------------------------------------
def scattering_length(Es, Zs, B):

    RL = larmor_radius(Es, Zs, B)

    if RL < lbd_coh:
        return (RL/lbd_coh)**(1/3) * lbd_coh # Mpc
    
    elif RL >= lbd_coh:
        return (RL/lbd_coh)**2 * lbd_coh # Mpc
    
# ----------------------------------------------------------------------------------------------------
def diffusive_distr(r, Es, cts, Zs, B):

    lbd_scatt = scattering_length(Es, Zs, B) 

    if r <= cts:
        sgm = np.sqrt(lbd_scatt * cts / 3)
        A = (sgm**2 * (np.sqrt(np.pi / 2) * sgm * erf(cts / (2*sgm)) - cts * np.exp(-cts**2 / (2 * sgm**2))))**-1
        return A * r**2 * np.exp(-r**2 / (2 * sgm**2))

    elif r > cts:
        return 0

# ----------------------------------------------------------------------------------------------------
def transition_distr(r, Es, cts, Zs, B):

    lbd_scatt = scattering_length(Es, Zs, B) 

    if r <= cts:
        alp = 3 * cts / lbd_scatt
        return r**2 * alp * np.exp(-alp / np.sqrt(1 - (r / cts)**2)) / (cts**3 * k1(alp) * (1 - (r / cts)**2)**2)

    elif r > cts:
        return 0

# ----------------------------------------------------------------------------------------------------
def w_mag(Es, cts, Zs, Dmin, Dmax, has_magnetic_field, B):

    if has_magnetic_field == False:
        if Dmin <= cts <= Dmax:
            return 1
        else:
            return 0

    elif has_magnetic_field == True:
        
        alp = 3 * cts / scattering_length(Es, Zs, B)

        if alp < 0.1:
            if Dmin <= cts <= Dmax:
                return 1
            else:
                return 0

        elif 0.1 <= alp <= 10:
            return quad(transition_distr, Dmin, Dmax, args = (Es, cts, Zs, B))[0]

        elif alp > 10:
            return quad(diffusive_distr, Dmin, Dmax, args = (Es, cts, Zs, B))[0]

# ----------------------------------------------------------------------------------------------------
def w_sim(Es, cts):

    return (Es * 1e18) * cts

# ----------------------------------------------------------------------------------------------------
def w_spec(Es, Zs, Rcut):

    return (Es * 1e18)**-Gmm * np.exp(-(Es * 1e18) / (Zs * Rcut))

# ----------------------------------------------------------------------------------------------------
def compute_spectrum(Zs, Rcut, Dmin, Dmax, has_magnetic_field, B):

    spec = np.zeros_like(ES) 

    for icts, cts in enumerate(CTSS):
        for iEs, Es in enumerate(ESS):

            data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZs(Zs)]}/S_ID{iZs(Zs):02d}D{icts:02d}E0{iEs:02d}.dat")

            spec += data * w_mag(Es, cts, Zs, Dmin, Dmax, has_magnetic_field, B) * w_sim(Es, cts) * w_spec(Es, Zs, Rcut)

    return spec

# ----------------------------------------------------------------------------------------------------
def get_EGMF_label(has_magnetic_field):

    if has_magnetic_field == False:
        return 'NoEGMF'

    elif has_magnetic_field == True:
        return 'EGMF'
    
# ----------------------------------------------------------------------------------------------------
def write_spectrum(Zs, Rcut, dist_arr, has_magnetic_field, B): 

    spec = []
    
    for idist in range(len(dist_arr) - 1): 
        spec.append(compute_spectrum(Zs, Rcut, dist_arr[idist], dist_arr[idist + 1], has_magnetic_field, B))
    
    if np.array_equal(dist_arr, [1, 3, 9, 27, 81, 243, 729, CTSS[-1]]):
        np.savetxt(f"{RESULTS_DIR}/spec_{PARTICLES[iZs(Zs)]}_{get_EGMF_label(has_magnetic_field)}.dat", np.column_stack((ES * 1e18, np.array(spec).T / (ES[:, np.newaxis] * 1e18))), fmt = "%.15e")
    else:
        np.savetxt(f"{RESULTS_DIR}/spec_{PARTICLES[iZs(Zs)]}_{get_EGMF_label(has_magnetic_field)}_Dmin_{int(dist_arr[0])}Mpc.dat", np.column_stack((ES * 1e18, np.array(spec).T / (ES[:, np.newaxis] * 1e18))), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # for Zs in ZSS:
    #     write_spectrum(Zs, 1e21, [1, 3, 9, 27, 81, 243, 729, CTSS[-1]], False, 0)
    #     write_spectrum(Zs, 1e21, [1, 3, 9, 27, 81, 243, 729, CTSS[-1]], True, 3)

    for Dmin in [3, 9, 27, 81, 243]:
        for Zs in ZSS:
            write_spectrum(Zs, 1e19, np.arange(Dmin, 10**3.5, Dmin), True, 1)

# ----------------------------------------------------------------------------------------------------