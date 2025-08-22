from scipy.integrate import quad
import numpy as np

RESULTS_DIR = "../results"

PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZSS = [1, 2, 7, 14, 26]

eV_to_erg = 1.60218e-12
EeV_to_eV = 1e18

Emin = 10**17.8 # eV
L0 = 5e44 # erg Mpc^-3 yr^-1
Gmm = -1.47
Rcut = 10**18.19 # V

E0 = EeV_to_eV 

# ----------------------------------------------------------------------------------------------------
def iZs(Zs):

    try:
        return ZSS.index(Zs)
    except ValueError:
        raise ValueError(f"Zs ({Zs}) not found in ZSS.")

# ----------------------------------------------------------------------------------------------------
def get_EGMF_label(has_magnetic_field):

    if has_magnetic_field == False:
        return 'NoEGMF'

    elif has_magnetic_field == True:
        return 'EGMF'

# ----------------------------------------------------------------------------------------------------
def integrand_w_spec(Es, Zs):

    if Es <= Zs * Rcut: 
        return Es * (Es / E0)**-Gmm
    else: 
        return Es * (Es / E0)**-Gmm * np.exp(1 - Es / (Zs * Rcut)) 

# ----------------------------------------------------------------------------------------------------
def normalize_spectrum(Zs, Dshell, has_magnetic_field):

    data = np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/spec_{PARTICLES[iZs(Zs)]}_{get_EGMF_label(has_magnetic_field)}.dat")
    return data[:,0], data[:, 1:] * [0.0, 0.245, 0.681, 0.049, 0.025][iZs(Zs)] * L0 / (quad(integrand_w_spec, Emin, 9e22, args = (Zs))[0] * eV_to_erg**2) / 1000

# ----------------------------------------------------------------------------------------------------
def write_spectrum(Zs, Dshell, has_magnetic_field):

    Es, spec = normalize_spectrum(Zs, Dshell, has_magnetic_field)
    mask = Es <= 10**20.5 # eV
    Es, spec = Es[mask], spec[mask]

    np.savetxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/norm_spec_{PARTICLES[iZs(Zs)]}_{get_EGMF_label(has_magnetic_field)}.dat", np.column_stack((Es, spec)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for Dshell in range(1, 10):
        for Zs in ZSS:
            write_spectrum(Zs, Dshell, True)

# ----------------------------------------------------------------------------------------------------