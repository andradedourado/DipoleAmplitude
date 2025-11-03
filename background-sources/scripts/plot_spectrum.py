from matplotlib.offsetbox import AnchoredText
from matplotlib.pylab import cm
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'legend.fontsize': 'x-large',
'legend.title_fontsize': 'x-large',
'axes.labelsize': 'xx-large',
'axes.titlesize': 'xx-large',
'xtick.labelsize': 'x-large',
'ytick.labelsize': 'x-large'})

FIGURES_DIR = "../figures"
RESULTS_DIR = "../results"

PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
PARTICLES_LEGEND = [r'$^1\mathrm{H}$', r'$^4\mathrm{He}$', r'$^{14}\mathrm{N}$', r'$^{28}\mathrm{Si}$', r'$^{56}\mathrm{Fe}$']
ZSS = [1, 2, 7, 14, 26]

# ----------------------------------------------------------------------------------------------------
def iZs(Zs):

    try:
        return ZSS.index(Zs)
    except ValueError:
        raise ValueError(f"Zs ({Zs}) not found in ZSS.")

# ----------------------------------------------------------------------------------------------------
def get_color(Zs):

    if Zs == 1:
        return cm.PuRd(np.linspace(0, 1, 10)[7])
    
    elif Zs == 2:
        return cm.BuGn(np.linspace(0, 1, 10)[7])
    
    elif Zs == 7:
        return cm.OrRd(np.linspace(0, 1, 10)[7])
    
    elif Zs == 14:
        return cm.GnBu(np.linspace(0, 1, 10)[7])
    
    elif Zs == 26:
        return cm.BuPu(np.linspace(0, 1, 10)[7])

# ----------------------------------------------------------------------------------------------------
def get_ylim(model):

    if model == 'CF2017':
        plt.ylim([5e41, 5e45])
    
    elif model == 'CF2023':
        plt.ylim([5e27, 5e31])

# ----------------------------------------------------------------------------------------------------
def number_density(Dshell):

    return 1e4 / (4 * np.pi * Dshell**3) # 1e-4 Mpc^-3

# ----------------------------------------------------------------------------------------------------
def plot_spectrum(Dshell, model):

    spec = np.zeros(len(np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/spec_1H_{model}_EGMF.dat")))

    for Zs in ZSS:
        data = np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/spec_{PARTICLES[iZs(Zs)]}_{model}_EGMF.dat")
        data = np.nan_to_num(data, nan = 0.0) # Check this later

        plt.plot(np.log10(data[:, 0]), data[:,0]**3 * np.sum(data[:, 1:], axis = 1), c = get_color(Zs), label = PARTICLES_LEGEND[iZs(Zs)])
        spec += data[:,0]**3 * np.sum(data[:, 1:], axis = 1)

        # y_smooth = savgol_filter(data[:,0]**3 * np.sum(data[:, 1:], axis = 1), window_length = 11, polyorder = 3)
        # plt.plot(np.log10(data[:, 0]), y_smooth, c = get_color(Zs), label = PARTICLES_LEGEND[iZs(Zs)])
        # spec += y_smooth

    plt.plot(np.log10(data[:, 0]), spec, c = 'k', label = 'All')
    plt.gca().add_artist(AnchoredText(r'$n = {:.2f} \times 10^{{-4}} \: \rm Mpc^{{-3}}$'.format(number_density(Dshell)), loc = 'upper left', frameon = False, prop = {'fontsize': 'x-large'}))
    plt.yscale('log')
    plt.xlim([18, 20.5])
    get_ylim(model)
    plt.xlabel(r'$\log_{10}(\rm Energy / eV)$')
    plt.ylabel(r'$E^3 \times {\rm Intensity} \: \rm [arb. units]$')
    plt.legend(title = r'$Z_s$')
    plt.savefig(f"{FIGURES_DIR}/spectrum_{int(Dshell)}Mpc_{model}.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/spectrum_{int(Dshell)}Mpc_{model}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for Dshell in range(1, 10):
        plot_spectrum(Dshell, 'CF2017')

# ----------------------------------------------------------------------------------------------------