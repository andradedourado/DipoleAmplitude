from scipy.integrate import quad, simps
from scipy.special import erf
from scipy.special import kn
import numpy as np

RESULTS_DIR = "../results"

ct_Hub = 4283 # Mpc

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
def d2N_dAdt_bal(lbd_scatt, rs): # Equation (5)

    return np.exp(- rs / lbd_scatt) / (4 * np.pi * rs**2)

# ----------------------------------------------------------------------------------------------------
def write_angular_distribution(lbd_scatt, rs, ilbd_scatt):

    theta = np.arccos(np.concatenate([-1 + np.arange(100) * 1.95 / 100, 0.95 + np.arange(100) * 0.05 / 100]))
    dN_dcos_theta = np.zeros_like(theta)

    for i in range(len(theta)):
        dN_dcos_theta[i] = quad(integrand_d2N_dAdt_diff, 0, np.inf, args = (theta[i], lbd_scatt, rs))[0]

    if ilbd_scatt == -1:
        dN_dcos_theta = dN_dcos_theta / (simps(dN_dcos_theta, np.cos(theta)) + d2N_dAdt_bal(lbd_scatt, rs)) # Normalization 
        lbd_scatt_over_rs_str = str(lbd_scatt / rs).rstrip('0').rstrip('.').replace('.', '_')
        np.savetxt(f"{RESULTS_DIR}/angular_distr_{rs}Mpc_{lbd_scatt_over_rs_str}.dat", np.column_stack((np.cos(theta), dN_dcos_theta)), fmt = "%.15e")

    else:
        np.savetxt(f"{RESULTS_DIR}/Figure3/angular_distr_{rs}Mpc_{ilbd_scatt:02d}.dat", np.column_stack((np.cos(theta), dN_dcos_theta)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    rs = 27 # Mpc 
    for lbd_scatt_over_rs in [0.01, 0.03, 0.1, 0.3, 1, 3]: 
        write_angular_distribution(lbd_scatt_over_rs * rs, rs, -1)

    for rs in [3, 27, 243]:
        for ilbd_scatt, lbd_scatt_over_rs in enumerate(np.logspace(-3, 3, num = 100)):
            write_angular_distribution(lbd_scatt_over_rs * rs, rs, ilbd_scatt)

# ----------------------------------------------------------------------------------------------------