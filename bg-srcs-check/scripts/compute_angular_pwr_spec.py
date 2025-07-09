import numpy as np

RESULTS_DIR = "../results"

# ----------------------------------------------------------------------------------------------------
def compute_angular_power_spectrum(l):

    lbd_scatt_over_rs = np.loadtxt(f"{RESULTS_DIR}/sph_harmonics_coeffs_27Mpc_{l:02d}.dat")[:,0]
    num = (np.loadtxt(f"{RESULTS_DIR}/sph_harmonics_coeffs_27Mpc_{l:02d}.dat")[:,1])**2 / (2 * l + 1)  

    den = np.zeros_like(num)

    for n in range(100):   
        den += (np.loadtxt(f"{RESULTS_DIR}/sph_harmonics_coeffs_27Mpc_{n:02d}.dat")[:,1])**2 / (2 * n + 1)

    return lbd_scatt_over_rs, num / den

# ----------------------------------------------------------------------------------------------------
def write_angular_power_spectrum():

    for l in range(5):
        lbd_scatt_over_rs, Cl = compute_angular_power_spectrum(l)
        np.savetxt(f"{RESULTS_DIR}/angular_pwr_spec_27Mpc_{l:02d}.dat", np.column_stack((lbd_scatt_over_rs, Cl)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    write_angular_power_spectrum()

# ----------------------------------------------------------------------------------------------------