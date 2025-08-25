from scipy.integrate import simps
from scipy.special import legendre
import numpy as np

RESULTS_DIR = "../results"
SIMULATIONS_DIR = "../../simulations/background-sources/results/angular-distr"

ES = np.logspace(4. / (2. * 79), 4. - 4. / (2. * 79), num = 79)
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZS = [1, 2, 7, 14, 26]

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
def d2N_dAdt_bal(E, rs, Z): # Equation (5)

    return np.exp(- rs / scattering_length(E, Z)) / (4 * np.pi * rs**2)

# ----------------------------------------------------------------------------------------------------
def compute_sph_harmonics_coeffs(l, iE, rs, Z):

    data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZ(Z)]}/{int(rs)}Mpc/angular_distr_{iE:02d}.dat")

    N = (simps(data[:,1], data[:,0]) + d2N_dAdt_bal(ES[iE], rs, Z)) / 2
    x, integrand = data[:,0], legendre(l)(data[:,0]) * data[:,1]        

    return (2*l + 1) / 2 / N * (simps(integrand, x) + d2N_dAdt_bal(ES[iE], rs, Z) * legendre(l)(1))

# ----------------------------------------------------------------------------------------------------
def Phi_l_tot(l, Dshell, Z):

    spec = np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/spec_{PARTICLES[iZ(Z)]}_EGMF.dat")

    Phi_l_tot = np.zeros_like(ES)

    for irs, rs in enumerate(np.arange(27, 860, Dshell)):
        
        if irs + 1 >= spec.shape[1]: # To understand
            break
        
        Phi_l = np.zeros_like(ES)
        
        for iE in range(len(ES)):
            Phi_l_temp = compute_sph_harmonics_coeffs(l, iE, rs, Z)
            if np.isfinite(Phi_l_temp):
                Phi_l[iE] = Phi_l_temp

        Phi_l_tot += (Phi_l * spec[:,irs+1] / (irs + 1))**2

    return spec[:,0], np.sqrt(Phi_l_tot)

# ----------------------------------------------------------------------------------------------------
def Phi_0_tot(Dshell, Z):

    spec = np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/spec_{PARTICLES[iZ(Z)]}_EGMF.dat")
    return spec[:,0], np.sum(spec[:, 1:], axis = 1)
    
# ----------------------------------------------------------------------------------------------------
def write_sph_coeffs_srcs(l, Dshell, Z):

    if l == 0:
        E, coeffs = Phi_0_tot(Dshell, Z)
        np.savetxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/Phi_{l}_tot_{PARTICLES[iZ(Z)]}.dat", np.column_stack((E, coeffs)), fmt = "%.15e")

    else:
        E, coeffs = Phi_l_tot(l, Dshell, Z)
        np.savetxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/Phi_{l}_tot_{PARTICLES[iZ(Z)]}.dat", np.column_stack((E, coeffs)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for Z in ZS:
        for Dshell in range(1, 10):
            write_sph_coeffs_srcs(0, Dshell, Z)
            write_sph_coeffs_srcs(1, Dshell, Z)

# ----------------------------------------------------------------------------------------------------