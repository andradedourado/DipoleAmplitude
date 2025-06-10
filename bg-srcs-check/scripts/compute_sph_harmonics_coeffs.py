from scipy.integrate import simps
from scipy.special import legendre
import numpy as np

RESULTS_DIR = "../results"

# ----------------------------------------------------------------------------------------------------
def integrand_sph_harmonics_coeffs(l, rs, ilbd_scatt):

    data = np.loadtxt(f"{RESULTS_DIR}/angular_distr_{ilbd_scatt:02d}_{rs}Mpc.dat")
    return data[:,0], legendre(l)(data[:,0]) * data[:,1]

# ----------------------------------------------------------------------------------------------------
def compute_sph_harmonics_coeffs(l, rs, ilbd_scatt):

    x, integrand = integrand_sph_harmonics_coeffs(l, rs, ilbd_scatt)
    return (2*l + 1) / 2 * simps(integrand, x)

# ----------------------------------------------------------------------------------------------------
def write_sph_harmonics_coeffs(l, lbd_scatt_over_rs, rs):

    Phi_l = np.zeros_like(lbd_scatt_over_rs)

    for ilbd_scatt in range(len(lbd_scatt_over_rs)):
        Phi_l[ilbd_scatt] = compute_sph_harmonics_coeffs(l, rs, ilbd_scatt)

    np.savetxt(f"{RESULTS_DIR}/sph_harmonics_coeffs_{l}_{rs}Mpc.dat", np.column_stack((lbd_scatt_over_rs, Phi_l)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    lbd_scatt_over_rs = np.logspace(-3, 3, num = 100)

    for rs in [3, 27, 243]:
        write_sph_harmonics_coeffs(1, lbd_scatt_over_rs, rs)

# ----------------------------------------------------------------------------------------------------