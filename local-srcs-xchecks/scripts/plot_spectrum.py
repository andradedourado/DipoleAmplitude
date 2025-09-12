from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
import numpy as np 

plt.rcParams.update({'legend.fontsize': 'medium',
'legend.title_fontsize': 'large',
'axes.labelsize': 'x-large',
'axes.titlesize': 'xx-large',
'xtick.labelsize': 'x-large',
'ytick.labelsize': 'x-large'})

RESULTS_DIR = "../results"

GALAXIES = ['CenA', 'ForA', 'VirA', \
            'NGC253', 'M82', 'NGC4945', 'M83', 'IC342', 'NGC6946', 'NGC2903', 'NGC5055', 'NGC3628', 'NGC3627', \
            'NGC4631', 'NGC891', 'NGC3556', 'NGC660', 'NGC2146', 'NGC3079', 'NGC1068', 'NGC1365']
PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
PARTICLES_LEGEND = [r'$^{1}$H', r'$^{4}$He', r'$^{14}$N', r'$^{28}$Si', r'$^{56}$Fe']
ZSS = [1, 2, 7, 14, 26]

# ----------------------------------------------------------------------------------------------------
def iZs(Zs):

    try:
        return ZSS.index(Zs)
    except ValueError:
        raise ValueError(f"Zs ({Zs}) not found in ZSS.")

# ----------------------------------------------------------------------------------------------------
def plot_spectrum(galaxy, Zs):

    data = np.loadtxt(f"{RESULTS_DIR}/spec_{galaxy}_{PARTICLES[iZs(Zs)]}.dat")
    plt.plot(np.log10(data[:,0]), data[:,1] * data[:,0]**2, color = 'k')

    plt.gca().add_artist(AnchoredText(f'{galaxy} | {PARTICLES_LEGEND[iZs(Zs)]}', loc = 'lower left', frameon = False, prop = {'fontsize': 'x-large'}))

    plt.yscale('log')
    plt.xlim([18, 21])
    plt.xlabel(r'$\log_{10}{({\rm Energy}/{\rm eV})}$')
    plt.ylabel(r'$E^2 \times {\rm Intensity} \: \rm [arb. units]$')
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for galaxy in GALAXIES:
        for Zs in ZSS:
            plot_spectrum(galaxy, Zs)

# ----------------------------------------------------------------------------------------------------