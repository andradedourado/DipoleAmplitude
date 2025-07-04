from scipy.integrate import quad, simps
from scipy.special import erf
from scipy.special import k1
import numpy as np

RESULTS_DIR = "../results"

ES = np.logspace(4. / (2. * 79), 4. - 4. / (2. * 79), num = 79)
ES = ES[ES <= 100]
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZS = [1, 2, 7, 14, 26]

ct_Hub = 10**3.5 # Mpc

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
def d3N_dr3_Juttner(cts, lbd_scatt, r):

    if r < cts:
        alp = 3 * cts / lbd_scatt
        return r**2 * alp * np.exp(-alp / np.sqrt(1 - (r / cts)**2)) / (cts**3 * k1(alp) * (1 - (r / cts)**2)**2) / (4 * np.pi * r**2)

    elif r >= cts:
        return 0.

# ----------------------------------------------------------------------------------------------------
def d3N_dr3_diff(cts, lbd_scatt, r):

    if r <= cts:
        sgm = np.sqrt(lbd_scatt * cts / 3)
        A = (sgm**2 * (np.sqrt(np.pi / 2) * sgm * erf(cts / (2*sgm))) - cts * np.exp(-cts**2 / (2 * sgm**2)))**-1
        return A * r**2 * np.exp(-r**2 / (2 * sgm**2)) / (4 * np.pi * r**2)

    elif r > cts:
        return 0.

# ----------------------------------------------------------------------------------------------------
def n(lbd_scatt, r): # Equation (3)

    if r < 0.1 * lbd_scatt / 3:
        return 1 / (4 * np.pi * r**2) + quad(d3N_dr3_Juttner, 0.1 * lbd_scatt / 3, 10 * lbd_scatt / 3, args = (lbd_scatt, r))[0] + quad(d3N_dr3_diff, 10 * lbd_scatt / 3, ct_Hub, args = (lbd_scatt, r))[0]
    
    else: 
        return quad(d3N_dr3_Juttner, 0.1 * lbd_scatt / 3, 10 * lbd_scatt / 3, args = (lbd_scatt, r))[0] + quad(d3N_dr3_diff, 10 * lbd_scatt / 3, ct_Hub, args = (lbd_scatt, r))[0]

# ----------------------------------------------------------------------------------------------------
def integrand_d2N_dAdt_diff(l, cos_theta, lbd_scatt, rs):

    if np.isclose(cos_theta, 1) == True:
        r = abs(rs - l)
    if np.isclose(cos_theta, -1) == True:
        r = rs + l
    else:
        r = np.sqrt(l**2 + rs**2 - 2 * l * rs * cos_theta)

    if np.isclose(r, 0) == True: # To prevent numerical issues
        return 0
    else:
        return np.exp(- l / lbd_scatt) * n(lbd_scatt, r) / lbd_scatt

# ----------------------------------------------------------------------------------------------------
def d2N_dAdt_diff(cos_theta, lbd_scatt, rs): # Equation (2)

    return quad(integrand_d2N_dAdt_diff, 0, np.inf, args = (cos_theta, lbd_scatt, rs))[0]

# ----------------------------------------------------------------------------------------------------
def write_angular_distribution(E, Z, rs, iE): 

    cos_theta = np.cos(np.linspace(np.pi, 0, num = 100))
    # cos_theta = np.concatenate([np.linspace(-1, 0.95, num = 100), np.linspace(0.95, 1, num = 100)[1:]])
    dN_dcos_theta = np.zeros_like(cos_theta)

    for i in range(len(cos_theta)):
        dN_dcos_theta[i] = d2N_dAdt_diff(cos_theta[i], scattering_length(E, Z), rs)

    np.savetxt(f"{RESULTS_DIR}/Figure4/{rs}Mpc/angular_distr_{PARTICLES[iZ(Z)]}_{iE:02d}.dat", np.column_stack((cos_theta, dN_dcos_theta)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for Dmin in [3, 9, 27, 81, 243]: # Mpc
        for iE, E in enumerate(ES): 
            write_angular_distribution(E, 1, Dmin, iE)

# ----------------------------------------------------------------------------------------------------