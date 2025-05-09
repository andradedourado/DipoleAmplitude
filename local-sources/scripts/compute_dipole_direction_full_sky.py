from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np

with open("simulations_dir.txt", "r") as f:
    SIMULATIONS_DIR = f.read().strip()

GALAXIES_DIR = "../../galaxies"
RESULTS_DIR = "../results"

ENERGY_EDGES = np.array([4., 8., 16., 32.])
ENERGY_BIN_CENTERS = np.array([6., 9., 24.]) # Modify
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
ZSS = [1, 2, 7, 14, 26]

Gmm = 1.22
Rcut = 10**18.72 # V

# ----------------------------------------------------------------------------------------------------
def iZs(Zs):

    try:
        return ZSS.index(Zs)
    except ValueError:
        raise ValueError(f"Zs ({Zs}) not found in ZSS.")

# ----------------------------------------------------------------------------------------------------
def w_L(galaxy, L):
    
    data = np.genfromtxt(f"{GALAXIES_DIR}/galaxies.dat", dtype = None, encoding = 'utf-8')

    if L == 'L11':
        iL = 4 
    elif L == 'Lradio':
        iL = 5 
    elif L == 'Lgamma':
        iL = 6

    for isrc in range(len(data)):
        if galaxy == data[isrc][0]:
            return data[isrc][iL]

# ----------------------------------------------------------------------------------------------------
def w_sim(Es):

    return (Es * 1e18)

# ----------------------------------------------------------------------------------------------------
def w_spec(Es, Zs):

    return (Es * 1e18)**-Gmm * np.exp(-(Es * 1e18) / (Zs * Rcut))

# ----------------------------------------------------------------------------------------------------
def w_Zs(Zs):

    return [6.4, 46.7, 37.5, 9.4, 0.0][iZs(Zs)]

# ----------------------------------------------------------------------------------------------------
def get_galaxy_set(galaxy_type):

    if galaxy_type == 'AGN':
        return ['CenA', 'ForA', 'VirA']
    
    elif galaxy_type == 'AGN+SBG':
        return ['CenA', 'ForA', 'VirA', \
            'NGC253', 'M82', 'NGC4945', 'M83', 'IC342', 'NGC6946', 'NGC2903', 'NGC5055', 'NGC3628', 'NGC3627', \
            'NGC4631', 'NGC891', 'NGC3556', 'NGC660', 'NGC2146', 'NGC3079', 'NGC1068', 'NGC1365']
    
    elif galaxy_type == 'SBG':
        return ['NGC253', 'M82', 'NGC4945', 'M83', 'IC342', 'NGC6946', 'NGC2903', 'NGC5055', 'NGC3628', 'NGC3627', \
            'NGC4631', 'NGC891', 'NGC3556', 'NGC660', 'NGC2146', 'NGC3079', 'NGC1068', 'NGC1365']

# ----------------------------------------------------------------------------------------------------
def compute_S0_Sx_Sy_Sz_binned(galaxy_type, L):

    S0, Sx, Sy, Sz = (np.zeros_like(ENERGY_BIN_CENTERS) for _ in range(4))

    for Zs in ZSS:
        for galaxy in get_galaxy_set(galaxy_type):
            for EGMF in range(20):
                data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZs(Zs)]}/events_{galaxy}_EGMF{EGMF:02d}.txt")
                weights = w_L(galaxy, L) * w_sim(data[:,10]) * w_spec(data[:,10], Zs) * w_Zs(Zs)
                
                S0 += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = weights)[0]
                Sx += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = -data[:,6] * weights)[0]
                Sy += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = -data[:,7] * weights)[0]
                Sz += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = -data[:,8] * weights)[0]

    return S0, Sx, Sy, Sz

# ----------------------------------------------------------------------------------------------------
def compute_S0_Sx_Sy_Sz_threshold(galaxy_type, L, E_thr):

    S0, Sx, Sy, Sz = (np.zeros_like(ENERGY_BIN_CENTERS) for _ in range(4))

    for Zs in ZSS:
        for galaxy in get_galaxy_set(galaxy_type):
            for EGMF in range(20):
                data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZs(Zs)]}/events_{galaxy}_EGMF{EGMF:02d}.txt")
                
                mask = data[:,2] > E_thr

                weights = w_L(galaxy, L) * w_sim(data[:,10]) * w_spec(data[:,10], Zs) * w_Zs(Zs)

                S0 += weights[mask]
                Sx += -data[:,6][mask] * weights[mask]
                Sy += -data[:,7][mask] * weights[mask]
                Sz += -data[:,8][mask] * weights[mask]

    return S0, Sx, Sy, Sz

# ----------------------------------------------------------------------------------------------------
def compute_dipole_direction(galaxy_type, L, dipole_kind, E_thr):

    if dipole_kind == 'binned':

        S0, Sx, Sy, Sz = compute_S0_Sx_Sy_Sz_binned(galaxy_type, L)

        D = np.array([Sx, Sy, Sz]) / np.sqrt(Sx**2 + Sy**2 + Sz**2)

        Dx, Dy, Dz = D[0], D[1], D[2]

        r = np.sqrt(Dx**2 + Dy**2 + Dz**2)
        theta = np.arccos(Dz / r) # [0°, 180°]
        phi = np.arctan2(Dy, Dx)  # [-180°, 180°]
        phi[phi < 0] += 2 * np.pi # [0°, 360°]

        sgb = np.pi / 2 - theta # [-90°, 90°]
        sgl = phi               # [0°, 360°]
        supergalactic_coords = SkyCoord(sgl = sgl * u.rad, sgb = sgb * u.rad, frame = 'supergalactic', unit = 'rad')

        galactic_coords = supergalactic_coords.galactic

        l = galactic_coords.l.to(u.rad).value
        b = galactic_coords.b.to(u.rad).value

        return l, b

    # elif dipole_kind == 'threshold':
    #     S0, Sx, Sy, Sz = compute_S0_Sx_Sy_Sz_threshold(galaxy_type, L, E_thr)

# ----------------------------------------------------------------------------------------------------
def write_dipole_direction(galaxy_type, L, dipole_kind, E_thr):

    l, b = compute_dipole_direction(galaxy_type, L, dipole_kind, E_thr)
    np.savetxt(f"{RESULTS_DIR}/dipole_direction_full_sky_{galaxy_type}_{L}.dat", np.column_stack((ENERGY_BIN_CENTERS * 1e18, l, b)), fmt = "%.15e", header = "Energy [eV]\tGalactic longitude [rad]\tGalactic latitude [rad]")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for galaxy_type in ['AGN', 'AGN+SBG', 'SBG']:
        for L in ['L11', 'Lradio', 'Lgamma']:
            write_dipole_direction(galaxy_type, L, 'binned', 0)

# ----------------------------------------------------------------------------------------------------