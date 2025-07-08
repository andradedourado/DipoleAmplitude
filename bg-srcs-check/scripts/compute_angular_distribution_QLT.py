from scipy.integrate import quad
from scipy.special import erf
from scipy.special import kn
import numpy as np

RESULTS_DIR = "../results"

ES = np.logspace(4. / (2. * 79), 4. - 4. / (2. * 79), num = 79)
ES = ES[ES <= 100]
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZS = [1, 2, 7, 14, 26]

ct_Hub = 4283 # Mpc

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
def law_of_cosines(l, theta, rs):

    if np.isclose(theta, 0) == True:
        return abs(rs - l)
    elif np.isclose(theta, np.pi) == True:
        return rs + l
    else:
        return np.sqrt(l**2 + rs**2 - 2 * l * rs * np.cos(theta))

# ----------------------------------------------------------------------------------------------------
def d3N_dr3_Juttner(cts, lbd_scatt, r):
    
    if r > cts:
        return 0.

    elif r <= cts:
        alp = 3 * cts / lbd_scatt
        return alp * np.exp(-alp / np.sqrt(1 - (r / cts)**2)) / (cts**3 * kn(1, alp) * (1 - (r / cts)**2)**2) / (4 * np.pi)

# ----------------------------------------------------------------------------------------------------
def d3N_dr3_diff(cts, lbd_scatt, r):

    if r > cts:
        return 0.

    elif r <= cts:
        sgm = np.sqrt(lbd_scatt * cts / 3)
        A = (sgm**2 * (np.sqrt(np.pi / 2) * sgm * erf(cts / (2*sgm)) - cts * np.exp(-cts**2 / (2 * sgm**2))))**-1
        return A * np.exp(-r**2 / (2 * sgm**2)) / (4 * np.pi)

# ----------------------------------------------------------------------------------------------------
def integrand_d2N_dAdt_diff(l, theta, lbd_scatt, rs):

    r = law_of_cosines(l, theta, rs)

    if r > 0 and r <= 0.1 * lbd_scatt / 3:
        return np.exp(- l / lbd_scatt) / lbd_scatt * (1 / (4 * np.pi * r**2) + quad(d3N_dr3_Juttner, 0.1 * lbd_scatt / 3, 10 * lbd_scatt / 3, args = (lbd_scatt, r))[0] + quad(d3N_dr3_diff, 10 * lbd_scatt / 3, ct_Hub, args = (lbd_scatt, r))[0])
    
    else: 
        return np.exp(- l / lbd_scatt) / lbd_scatt * (quad(d3N_dr3_Juttner, 0.1 * lbd_scatt / 3, 10 * lbd_scatt / 3, args = (lbd_scatt, r))[0] + quad(d3N_dr3_diff, 10 * lbd_scatt / 3, ct_Hub, args = (lbd_scatt, r))[0])

# ----------------------------------------------------------------------------------------------------
def write_angular_distribution(E, Z, rs, iE): 

    theta = np.arccos(np.concatenate([-1 + np.arange(100) * 1.95 / 100, 0.95 + np.arange(100) * 0.05 / 100]))
    dN_dcos_theta = np.zeros_like(theta)

    for i in range(len(theta)):
        dN_dcos_theta[i] = quad(integrand_d2N_dAdt_diff, 0, np.inf, args = (theta[i], scattering_length(E, Z), rs))[0]

    np.savetxt(f"{RESULTS_DIR}/Figure4/{rs}Mpc/angular_distr_{PARTICLES[iZ(Z)]}_{iE:02d}.dat", np.column_stack((np.cos(theta), dN_dcos_theta)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for Dmin in [3, 9, 27, 81, 243]: # Mpc
        for iE, E in enumerate(ES): 
            write_angular_distribution(E, 1, Dmin, iE)

# ----------------------------------------------------------------------------------------------------