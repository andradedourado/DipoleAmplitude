from scipy.integrate import simps
from scipy.special import eval_legendre
import numpy as np

RESULTS_DIR = "../results"

lbd_scatt_over_rs = np.logspace(-3, 3, num = 100)

# ----------------------------------------------------------------------------------------------------
def d2N_dAdt_bal(lbd_scatt, rs): # Equation (5)

    return np.exp(- rs / lbd_scatt) / (4 * np.pi * rs**2)

# ----------------------------------------------------------------------------------------------------
def compute_sph_harmonics_coeffs(l, rs, ilbd_scatt):

    data = np.loadtxt(f"{RESULTS_DIR}/Figure3/angular_distr_{rs}Mpc_{ilbd_scatt:02d}.dat")
    N = (simps(data[:,1], data[:,0]) + d2N_dAdt_bal(lbd_scatt_over_rs[ilbd_scatt] * rs, rs)) / 2.

    x, integrand = data[:,0], eval_legendre(l, data[:,0]) * data[:,1]
    return (2.*l + 1.) / 2. / N * (simps(integrand, x) + d2N_dAdt_bal(lbd_scatt_over_rs[ilbd_scatt] * rs, rs) * eval_legendre(l, 1))

# ----------------------------------------------------------------------------------------------------
def write_sph_harmonics_coeffs(l, rs):

    Phi_l = np.zeros_like(lbd_scatt_over_rs)

    for ilbd_scatt in range(len(lbd_scatt_over_rs)):
        Phi_l[ilbd_scatt] = compute_sph_harmonics_coeffs(l, rs, ilbd_scatt)

    np.savetxt(f"{RESULTS_DIR}/sph_harmonics_coeffs_{rs}Mpc_{l:02d}.dat", np.column_stack((lbd_scatt_over_rs, Phi_l)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for rs in [3, 27, 243]:
        write_sph_harmonics_coeffs(1, rs)

    for l in range(100):
        write_sph_harmonics_coeffs(l, 27)

# ----------------------------------------------------------------------------------------------------