from scipy.integrate import quad, simps
from scipy.special import erf
from scipy.special import k1
import numpy as np

RESULTS_DIR = "../results"

ct_Hub = 10**3.5 # Mpc

# ----------------------------------------------------------------------------------------------------
def d3N_dr3_Juttner(cts, lbd_scatt, r):

    if r <= cts:
        alp = 3 * cts / lbd_scatt
        return r**2 * alp * np.exp(-alp / np.sqrt(1 - (r / cts)**2)) / (cts**3 * k1(alp) * (1 - (r / cts)**2)**2) / (4 * np.pi * r**2)

    elif r > cts:
        return 0

# ----------------------------------------------------------------------------------------------------
def d3N_dr3_diff(cts, lbd_scatt, r):

    if r <= cts:
        sgm = np.sqrt(lbd_scatt * cts / 3)
        A = (sgm**2 * (np.sqrt(np.pi / 2) * sgm * erf(cts / (2*sgm))) - cts * np.exp(-cts**2 / (2 * sgm**2)))**-1
        return A * r**2 * np.exp(-r**2 / (2 * sgm**2)) / (4 * np.pi * r**2)

    elif r > cts:
        return 0

# ----------------------------------------------------------------------------------------------------
def n(lbd_scatt, r): # Equation (3)

    if r < 0.1 * lbd_scatt / 3:
        return 1 / (4 * np.pi * r**2) + quad(d3N_dr3_Juttner, 0.1 * lbd_scatt / 3, 10 * lbd_scatt / 3, args = (lbd_scatt, r))[0] + quad(d3N_dr3_diff, 10 * lbd_scatt / 3, ct_Hub, args = (lbd_scatt, r))[0]
    
    else: 
        return quad(d3N_dr3_Juttner, 0.1 * lbd_scatt / 3, 10 * lbd_scatt / 3, args = (lbd_scatt, r))[0] + quad(d3N_dr3_diff, 10 * lbd_scatt / 3, ct_Hub, args = (lbd_scatt, r))[0]

# ----------------------------------------------------------------------------------------------------
def integrand_d2N_dAdt_diff(l, cos_theta, lbd_scatt, rs):

    r = np.sqrt(l**2 + rs**2 - 2 * l * rs * cos_theta)

    return np.exp(- l / lbd_scatt) * n(lbd_scatt, r) / lbd_scatt

# ----------------------------------------------------------------------------------------------------
def d2N_dAdt_diff(cos_theta, lbd_scatt, rs): # Equation (2)

    return quad(integrand_d2N_dAdt_diff, 0, np.inf, args = (cos_theta, lbd_scatt, rs))[0]

# ----------------------------------------------------------------------------------------------------
def d2N_dAdt_bal(lbd_scatt, rs): # Equation (5)

    return np.exp(- rs / lbd_scatt) / (4 * np.pi * rs**2)

# ----------------------------------------------------------------------------------------------------
def compute_angular_distribution(cos_theta, lbd_scatt, rs): # Equation (6) 

    if np.isclose(cos_theta, 1) == True:
        return d2N_dAdt_bal(lbd_scatt, rs) + d2N_dAdt_diff(cos_theta, lbd_scatt, rs)
    
    else: 
        return d2N_dAdt_diff(cos_theta, lbd_scatt, rs)

# ----------------------------------------------------------------------------------------------------
def write_angular_distribution(lbd_scatt, rs, ilbd_scatt):

    cos_theta = np.cos(np.linspace(np.pi, 0, num = 100))
    dN_dcos_theta = np.zeros_like(cos_theta)

    for i in range(len(cos_theta)):
        dN_dcos_theta[i] = compute_angular_distribution(cos_theta[i], lbd_scatt, rs)

    dN_dcos_theta = dN_dcos_theta / simps(dN_dcos_theta, cos_theta) # Normalization 

    if ilbd_scatt == -1:
        lbd_scatt_over_rs_str = str(lbd_scatt / rs).rstrip('0').rstrip('.').replace('.', '_')
        np.savetxt(f"{RESULTS_DIR}/angular_distr_{lbd_scatt_over_rs_str}_{rs}Mpc.dat", np.column_stack((cos_theta, dN_dcos_theta)), fmt = "%.15e")

    else:
        np.savetxt(f"{RESULTS_DIR}/angular_distr_{ilbd_scatt:02d}_{rs}Mpc.dat", np.column_stack((cos_theta, dN_dcos_theta)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # rs = 27 # Mpc 
    # for lbd_scatt_over_rs in [0.01, 0.03, 0.1, 0.3, 1, 3]: 
    #     write_angular_distribution(lbd_scatt_over_rs * rs, rs, -1)

    for rs in [3, 27, 243]:
        for ilbd_scatt, lbd_scatt_over_rs in enumerate(np.logspace(-3, 3, num = 100)):
            write_angular_distribution(lbd_scatt_over_rs * rs, rs, ilbd_scatt)

# ----------------------------------------------------------------------------------------------------