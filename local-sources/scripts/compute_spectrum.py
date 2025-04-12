import numpy as np 

with open("simulations_dir.txt", "r") as f:
    SIMULATIONS_DIR = f.read().strip()

RESULTS_DIR = "../results"

ESS = np.delete(np.logspace(0, 4, num = 81), 0)
ES = np.logspace(4./(2.*(len(ESS) - 1)), 4. - 4./(2.*(len(ESS) - 1)), num = len(ESS) - 1)
GALAXIES = ['CenA', 'ForA', 'VirA', \
            'NGC253', 'M82', 'NGC4945', 'M83', 'IC342', 'NGC6946', 'NGC2903', 'NGC5055', 'NGC3628', 'NGC3627', \
            'NGC4631', 'NGC891', 'NGC3556', 'NGC660', 'NGC2146', 'NGC3079', 'NGC1068', 'NGC1365']
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
def w_sim(Es):

    return (Es * 1e18)

# ----------------------------------------------------------------------------------------------------
def w_spec(Es, Zs):

    return (Es * 1e18)**-Gmm * np.exp(-(Es * 1e18) / (Zs * Rcut))

# ----------------------------------------------------------------------------------------------------
def compute_spectrum(EGMF, galaxy, Zs):

    data = np.loadtxt(f"{SIMULATIONS_DIR}/{PARTICLES[iZs(Zs)]}/events_{galaxy}_EGMF{EGMF:02d}.txt")
    hist, bin_edges = np.histogram(data[:,2], bins = np.logspace(0, 4, num = 80), weights = w_sim(data[:,10]) * w_spec(data[:,10], Zs))
    return hist

# ----------------------------------------------------------------------------------------------------
def write_spectrum(galaxy, Zs):

    spec = np.zeros_like(ES)

    for EGMF in range(20):
        spec += compute_spectrum(EGMF, galaxy, Zs)  

    np.savetxt(f"{RESULTS_DIR}/spec_{galaxy}_{PARTICLES[iZs(Zs)]}.dat", np.column_stack((ES * 1e18, spec / (ES * 1e18))), fmt = "%.15e")
  
# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for galaxy in GALAXIES:
        for Zs in ZSS:
            write_spectrum(galaxy, Zs)

# ----------------------------------------------------------------------------------------------------