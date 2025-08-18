import healpy as hp
import numpy as np

RESULTS_DIR = "../results"
SIMULATIONS_DIR = "../../simulations/local-sources/events"

PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZS = [1, 2, 7, 14, 26]

EeV_to_eV = 1e18

Gmm = -1.47
Rcut = 10**18.19 # V

E0 = EeV_to_eV

# ----------------------------------------------------------------------------------------------------
def iZ(Z):

    try:
        return ZS.index(Z)
    except ValueError:
        raise ValueError(f"Z ({Z}) not found in ZS.")

# ----------------------------------------------------------------------------------------------------
def w_sim(E):

    return E * EeV_to_eV

# ----------------------------------------------------------------------------------------------------
def w_spec(E, Z):

    E = E * EeV_to_eV
    Ecut = Z * Rcut

    w = np.zeros_like(E)

    mask_low = E <= Ecut
    w[mask_low] = (E[mask_low] / E0) ** -Gmm

    mask_high = E > Ecut
    w[mask_high] = (E[mask_high] / E0) ** -Gmm * np.exp(1 - E[mask_high] / Ecut)

    return w

# ----------------------------------------------------------------------------------------------------
def cartesian_to_spherical(Px, Py, Pz): 

    r = np.sqrt(Px**2 + Py**2 + Pz**2)
    tht = np.arccos(Pz / r) # Colatitude
    phi = np.arctan2(Py, Px)
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)

    return tht, phi

# ----------------------------------------------------------------------------------------------------
def generate_density_maps(Z, galaxy, EGMF, nside):

    data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZ(Z)]}/events_{galaxy}_EGMF{EGMF:02d}.txt")

    E = data[:,2]
    Px = -data[:,6]
    Py = -data[:,7]
    Pz = -data[:,8]

    weights = w_sim(E) * w_spec(E, Z)

    tht, phi = cartesian_to_spherical(Px, Py, Pz)

    npix = hp.nside2npix(nside)
    density_map = np.zeros(npix)
    pixels = hp.ang2pix(nside, tht, phi)
    density_map = np.bincount(pixels, weights = weights, minlength = npix)

    return density_map

# ----------------------------------------------------------------------------------------------------
def write_density_maps(Z, galaxy, EGMF, nside):

    np.savetxt(f"{RESULTS_DIR}/density_map_{PARTICLES[iZ(Z)]}_{galaxy}_EGMF{EGMF:02d}_{nside:02d}.dat", generate_density_maps(Z, galaxy, EGMF, nside), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for EGMF in range(20):
        write_density_maps(1, 'CenA', EGMF, 64)

# ----------------------------------------------------------------------------------------------------