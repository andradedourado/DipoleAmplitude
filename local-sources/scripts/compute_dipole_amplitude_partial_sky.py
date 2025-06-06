from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np

with open("simulations_dir.txt", "r") as f:
    SIMULATIONS_DIR = f.read().strip()

GALAXIES_DIR = "../../galaxies"
RESULTS_DIR = "../results"

ENERGY_EDGES = np.logspace(0, 4, num = 80)
ENERGY_BIN_CENTERS = np.logspace(4./(2.*(len(ENERGY_EDGES) - 1)), 4. - 4./(2.*(len(ENERGY_EDGES) - 1)), num = len(ENERGY_EDGES) - 1)
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
def convert_units(Px, Py, Pz):
    
    Px, Py, Pz = -Px, -Py, -Pz

    # Supergalactic spherical coordinates 
    r = np.sqrt(Px**2 + Py**2 + Pz**2) # The direction vector in CRPropa is already a unit vector
    theta = np.arccos(Pz / r) # [0°, 180°]
    phi = np.arctan2(Py, Px)  # [-180°, 180°]
    phi[phi < 0] += 2 * np.pi # [0°, 360°]

    # Supergalactic latitude and longitude 
    sgb = np.pi / 2 - theta # [-90°, 90°]
    sgl = phi               # [0°, 360°]
    supergalactic_coords = SkyCoord(sgl = sgl * u.rad, sgb = sgb * u.rad, frame = 'supergalactic', unit = 'rad')

    # Equatorial coordinates
    equatorial_coords = supergalactic_coords.icrs
    dec = equatorial_coords.dec.to(u.rad).value # [-90°, 90°]
    ra = equatorial_coords.ra.to(u.rad).value   # [0°, 360°]

    # Spherical coordinates: polar angle measured from the North Pole
    theta = np.pi / 2 - dec # [0°, 180°]  
    phi = ra                # [0°, 360°]

    Px = np.sin(theta) * np.cos(phi)
    Py = np.sin(theta) * np.sin(phi)
    Pz = np.cos(theta)

    return Px, Py, Pz

# ----------------------------------------------------------------------------------------------------
def compute_S0_Sx_Sy_Sz(galaxy_type, L):

    S0, Sx, Sy, Sz = (np.zeros_like(ENERGY_BIN_CENTERS) for _ in range(4))

    for Zs in ZSS:
        for galaxy in get_galaxy_set(galaxy_type):
            for EGMF in range(20):
                data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZs(Zs)]}/events_{galaxy}_EGMF{EGMF:02d}.txt")
                weights = w_L(galaxy, L) * w_sim(data[:,10]) * w_spec(data[:,10], Zs) * w_Zs(Zs)

                Px, Py, Pz = convert_units(data[:,6], data[:,7], data[:,8])
                
                S0 += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = weights)[0]
                Sx += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = Px * weights)[0]
                Sy += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = Py * weights)[0]
                Sz += np.histogram(data[:,2], bins = ENERGY_EDGES, weights = Pz * weights)[0]

    return S0, Sx, Sy, Sz

# ----------------------------------------------------------------------------------------------------
def compute_dipole_amplitude(galaxy_type, L):

    S0, Sx, Sy, Sz = compute_S0_Sx_Sy_Sz(galaxy_type, L)

    # Equatorial coordinates 
    dlt_min = -90
    dlt_max = 45

    # Spherical coordinates 
    tht_min = np.radians(90 - abs(dlt_min)) 
    tht_max = np.radians(90 - abs(dlt_max))

    # d = np.cos(tht_min) - np.cos(tht_max)
    s = np.cos(tht_min) + np.cos(tht_max)
    p = np.cos(tht_min) * np.cos(tht_max)

    gmm = (s**2 - p) / 3

    return np.sqrt((((gmm - p) / (gmm - 1))**2 * (Sx**2 + Sy**2) + (s * S0 - 2 * Sz)**2) / (2 * gmm * S0 - s * Sz)**2)

# ----------------------------------------------------------------------------------------------------
def write_dipole_amplitude(galaxy_type, L):

    alp_rec = compute_dipole_amplitude(galaxy_type, L)
    np.savetxt(f"{RESULTS_DIR}/dipole_amplitude_partial_sky_{galaxy_type}_{L}.dat", np.column_stack((ENERGY_BIN_CENTERS * 1e18, alp_rec)), fmt = "%.15e")

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for galaxy_type in ['AGN', 'AGN+SBG', 'SBG']:
        for L in ['L11', 'Lradio', 'Lgamma']:
            write_dipole_amplitude(galaxy_type, L)

# ----------------------------------------------------------------------------------------------------