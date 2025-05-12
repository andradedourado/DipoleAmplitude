from scipy.integrate import quad, simps
from scipy.special import erf
from scipy.special import k1
import numpy as np

RESULTS_DIR = "../results"

ct_Hub = (299792458) * (4.55 * 1.e17) * (3.24078 * 1.e-23) # 10**3.5 # Mpc

# ----------------------------------------------------------------------------------------------------
def d3N_dr3(cts, lbd_scatt, r):
    
    alp = 3 * cts / lbd_scatt

    if 0.1 <= alp <= 10:
        if r <= cts:
            return r**2 * alp * np.exp(-alp / np.sqrt(1 - (r / cts)**2)) / (cts**3 * k1(alp) * (1 - (r / cts)**2)**2) / (4 * np.pi * r**2)

        elif r > cts:
            return 0

    elif alp > 10:
        if r <= cts:
            sgm = np.sqrt(lbd_scatt * cts / 3)
            A = (sgm**2 * (np.sqrt(np.pi / 2) * sgm * erf(cts / (2*sgm))) - cts * np.exp(-cts**2 / (2 * sgm**2)))**-1
            return A * r**2 * np.exp(-r**2 / (2 * sgm**2)) / (4 * np.pi * r**2)

        elif r > cts:
            return 0

# ----------------------------------------------------------------------------------------------------
def n(lbd_scatt, r): # Equation (3)

    if r < 0.1 * lbd_scatt / 3:
        return 1 / (4 * np.pi * r**2) + quad(d3N_dr3, 0.1 * lbd_scatt / 3, ct_Hub, args = (lbd_scatt, r))[0]
    
    else: 
        return quad(d3N_dr3, 0.1 * lbd_scatt / 3, ct_Hub, args = (lbd_scatt, r))[0]

# ----------------------------------------------------------------------------------------------------
def integrand_d2N_dAdt_diff(l, cos_theta,lbd_scatt, r_s):

    r = np.sqrt(l**2 + r_s**2 - 2 * l * r_s * cos_theta)

    return np.exp(- l / lbd_scatt) * n(lbd_scatt, r) / lbd_scatt

# ----------------------------------------------------------------------------------------------------
def d2N_dAdt_diff(cos_theta,lbd_scatt, r_s): # Equation (2)

    d2N_dAdt_diff = np.zeros_like(cos_theta)

    for icos in range(len(cos_theta)):
        d2N_dAdt_diff[icos] = quad(integrand_d2N_dAdt_diff, 0, np.inf, args = (cos_theta[icos], lbd_scatt, r_s))[0]

    return d2N_dAdt_diff

# ----------------------------------------------------------------------------------------------------
def d2N_dAdt_bal(cos_theta,lbd_scatt, r_s): # Equation (5)

    mask = np.isclose(cos_theta, 1)

    d2N_dAdt_bal = np.zeros_like(cos_theta)
    d2N_dAdt_bal[mask] = np.exp(- r_s / lbd_scatt) / (4 * np.pi * r_s**2)

    return d2N_dAdt_bal

# ----------------------------------------------------------------------------------------------------
def compute_angular_distribution(cos_theta,lbd_scatt, r_s): # Equation (6)

    return d2N_dAdt_bal(cos_theta,lbd_scatt, r_s) + d2N_dAdt_diff(cos_theta,lbd_scatt, r_s)

# ----------------------------------------------------------------------------------------------------
def write_angular_distribution(lbd_scatt, r_s):

    cos_theta = np.cos(np.linspace(np.pi, 0, num = 100))
    dN_dcos_theta = compute_angular_distribution(cos_theta,lbd_scatt, r_s) 
    dN_dcos_theta = 2 * dN_dcos_theta / simps(dN_dcos_theta, cos_theta) # Normalization

    lbd_scatt_over_rs_str = str(lbd_scatt / r_s).rstrip('0').rstrip('.').replace('.', '_')

    np.savetxt(f"{RESULTS_DIR}/angular_distr_{lbd_scatt_over_rs_str}.dat", np.column_stack((cos_theta, dN_dcos_theta)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    r_s = 27 # Mpc 

    for lbd_scatt_over_rs in [0.01, 0.03, 0.1, 0.3, 1, 3]: 
        write_angular_distribution(lbd_scatt_over_rs * r_s, r_s)

# ----------------------------------------------------------------------------------------------------