from scipy.integrate import quad
from scipy.special import erf
from scipy.special import k1
import numpy as np 

RESULTS_DIR = "../results"
SIMULATIONS_DIR = "../../simulations/background-sources/results/binning"

CTSS = np.logspace(0, 3.5, num = 71)
ESS = np.delete(np.logspace(0, 4, num = 81), 0)
ES = np.logspace(4./(2.*(len(ESS) - 1)), 4. - 4./(2.*(len(ESS) - 1)), num = len(ESS) - 1)
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZSS = [1, 2, 7, 14, 26]

EeV_to_eV = 1e18
eV_to_erg = 1.60218e-12

Emin = 10**17.8 # eV

E0 = EeV_to_eV

B = 1 # nG 
lbd_coh = 1 # Mpc

# ----------------------------------------------------------------------------------------------------
def iZs(Zs):

    try:
        return ZSS.index(Zs)
    except ValueError:
        raise ValueError(f"Zs ({Zs}) not found in ZSS.")

# ----------------------------------------------------------------------------------------------------
def larmor_radius(Es, Zs): # E in EeV and B in nG 

    return 3.5 * 1.081 / Zs * Es / B # Mpc

# ----------------------------------------------------------------------------------------------------
def scattering_length(Es, Zs):

    RL = larmor_radius(Es, Zs)

    if RL < lbd_coh:
        return (RL/lbd_coh)**(1/3) * lbd_coh # Mpc
    
    elif RL >= lbd_coh:
        return (RL/lbd_coh)**2 * lbd_coh # Mpc
    
# ----------------------------------------------------------------------------------------------------
def diffusive_distr(r, Es, cts, Zs):

    lbd_scatt = scattering_length(Es, Zs) 

    if r <= cts:
        sgm = np.sqrt(lbd_scatt * cts / 3)
        A = (sgm**2 * (np.sqrt(np.pi / 2) * sgm * erf(cts / (2*sgm)) - cts * np.exp(-cts**2 / (2 * sgm**2))))**-1
        return A * r**2 * np.exp(-r**2 / (2 * sgm**2))

    elif r > cts:
        return 0

# ----------------------------------------------------------------------------------------------------
def transition_distr(r, Es, cts, Zs):

    lbd_scatt = scattering_length(Es, Zs) 

    if r <= cts:
        alp = 3 * cts / lbd_scatt
        return r**2 * alp * np.exp(-alp / np.sqrt(1 - (r / cts)**2)) / (cts**3 * k1(alp) * (1 - (r / cts)**2)**2)

    elif r > cts:
        return 0

# ----------------------------------------------------------------------------------------------------
def w_mag(Es, cts, Zs, Dmin, Dmax, has_magnetic_field):

    if has_magnetic_field == False:
        if Dmin <= cts <= Dmax:
            return 1
        else:
            return 0

    elif has_magnetic_field == True:
        
        alp = 3 * cts / scattering_length(Es, Zs)

        if alp < 0.1:
            if Dmin <= cts <= Dmax:
                return 1
            else:
                return 0

        elif 0.1 <= alp <= 10:
            return quad(transition_distr, Dmin, Dmax, args = (Es, cts, Zs))[0]

        elif alp > 10:
            return quad(diffusive_distr, Dmin, Dmax, args = (Es, cts, Zs))[0]

# ----------------------------------------------------------------------------------------------------
def w_sim(Es, cts):

    return (Es * EeV_to_eV) / E0 * cts

# ----------------------------------------------------------------------------------------------------
def w_spec(Es, Zs):

    L0, Gmm, Rcut = generation_rate_parameters(Zs)

    Es = Es * EeV_to_eV

    if Es <= Zs * Rcut:
        w_spec = (Es / E0)**-Gmm
    else:
        w_spec = (Es / E0)**-Gmm * np.exp(1 - Es / (Zs * Rcut))

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
def compute_spectrum(Zs, Dmin, Dmax, has_magnetic_field):

    spec = np.zeros_like(ES) 

    for icts, cts in enumerate(CTSS):
        for iEs, Es in enumerate(ESS):

            data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZs(Zs)]}/S_ID{iZs(Zs):02d}D{icts:02d}E0{iEs:02d}.dat")

            spec += data * w_mag(Es, cts, Zs, Dmin, Dmax, has_magnetic_field) * w_sim(Es, cts) * w_spec(Es, Zs)

    return spec

# ----------------------------------------------------------------------------------------------------
def get_EGMF_label(has_magnetic_field):

    if has_magnetic_field == False:
        return 'NoEGMF'

    elif has_magnetic_field == True:
        return 'EGMF'
    
# ----------------------------------------------------------------------------------------------------
def write_spectrum(Zs, Dshell, has_magnetic_field):

    dist_arr = np.arange(27, 3163, Dshell) # Mpc 

    spec = []
    
    for idist in range(len(dist_arr) - 1):
        print(Zs, dist_arr[idist])
        spec.append(compute_spectrum(Zs, dist_arr[idist], dist_arr[idist + 1], has_magnetic_field))
    
    np.savetxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/spec_{PARTICLES[iZs(Zs)]}_{get_EGMF_label(has_magnetic_field)}.dat", np.column_stack((ES * 1e18, np.array(spec).T / (ES[:, np.newaxis] * 1e18))), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for Dshell in range(1, 10):
        write_spectrum(26, Dshell, True) # Manually change the primary to run the script

# ----------------------------------------------------------------------------------------------------