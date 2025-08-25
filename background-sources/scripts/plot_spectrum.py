from scipy.signal import savgol_filter
from matplotlib.pylab import cm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'legend.fontsize': 'medium',
'legend.title_fontsize': 'large',
'axes.labelsize': 'x-large',
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
def plot_spectrum(Dshell):

    spec = np.zeros(len(np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/norm_spec_4He_EGMF.dat")))

    for Zs in ZSS[1:]:
        data = np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/norm_spec_{PARTICLES[iZs(Zs)]}_EGMF.dat")    
        y_smooth = savgol_filter(data[:,0]**3 * np.sum(data[:, 1:], axis = 1), window_length = 11, polyorder = 3)
        plt.plot(np.log10(data[:, 0]), y_smooth, c = get_color(Zs), label = PARTICLES_LEGEND[iZs(Zs)])
        # plt.plot(np.log10(data[:, 0]), data[:,0]**3 * np.sum(data[:, 1:], axis = 1), c = get_color(Zs), label = PARTICLES_LEGEND[iZs(Zs)])
        spec += y_smooth

    plt.plot(np.log10(data[:, 0]), spec, c = 'k', label = 'All')

    plt.yscale('log')
    plt.xlim([18, 20.5])
    plt.ylim([5e88, 5e90])
    plt.xlabel(r'$\log_{10}(\rm Energy / eV)$')
    plt.ylabel(r'$E^3 \times {\rm Intensity} \: \rm [arb. units]$')
    plt.legend(title = r'$Z_s$')
    # plt.savefig(f"{FIGURES_DIR}/spectrum_{int(Dshell)}Mpc.pdf", bbox_inches = 'tight')
    # plt.savefig(f"{FIGURES_DIR}/spectrum_{int(Dshell)}Mpc.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for Dshell in range(1, 10):
        plot_spectrum(Dshell)

# ----------------------------------------------------------------------------------------------------