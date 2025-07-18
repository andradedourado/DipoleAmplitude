from scipy.integrate import simps
from scipy.special import legendre
import numpy as np

RESULTS_DIR = "../results"

ES = np.logspace(4. / (2. * 79), 4. - 4. / (2. * 79), num = 79)
ES = ES[ES <= 100]
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZS = [1, 2, 7, 14, 26]

Z = 1

B = 1 # nG 
lbd_coh = 1 # Mpc

# ----------------------------------------------------------------------------------------------------
def iZ(Z):

    try:
        return ZS.index(Z)
    except ValueError:
        raise ValueError(f"Z ({Z}) not found in ZS.")

# ----------------------------------------------------------------------------------------------------
def scattering_length(E, Z):

    RL = 1.081 / Z * E / B # Mpc; E in EeV and B in nG 

    if RL < lbd_coh:
        return (RL/lbd_coh)**(1/3) * lbd_coh # Mpc
    
    else:
        return (RL/lbd_coh)**2 * lbd_coh # Mpc

# ----------------------------------------------------------------------------------------------------
def d2N_dAdt_bal(E, rs): # Equation (5)

    return np.exp(- rs / scattering_length(E, Z)) / (4 * np.pi * rs**2)

# ----------------------------------------------------------------------------------------------------
def compute_sph_harmonics_coeffs(l, rs, iE):

    data = np.loadtxt(f"{RESULTS_DIR}/Figure4/{int(rs)}Mpc/angular_distr_{PARTICLES[iZ(Z)]}_{iE:02d}.dat")
    N = (simps(data[:,1], data[:,0]) + d2N_dAdt_bal(ES[iE], rs)) / 2

    x, integrand = data[:,0], legendre(l)(data[:,0]) * data[:,1]
    return (2*l + 1) / 2 / N * (simps(integrand, x) + d2N_dAdt_bal(ES[iE], rs) * legendre(l)(1))

# ----------------------------------------------------------------------------------------------------
def write_sph_harmonics_coeffs(l, rs):

    Phi_l = np.zeros_like(ES)

    for iE in range(len(ES)):
        Phi_l[iE] = compute_sph_harmonics_coeffs(l, rs, iE)

    np.savetxt(f"{RESULTS_DIR}/sph_harmonics_coeffs_QLT_{int(rs)}Mpc_{l:02d}.dat", np.column_stack((ES * 1e18, Phi_l)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for rs in [3, 9, 27, 81, 243]:
        write_sph_harmonics_coeffs(1, rs)

# ----------------------------------------------------------------------------------------------------