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
ZSS = [1, 2, 7, 14, 26]

# ----------------------------------------------------------------------------------------------------
def iZs(Zs):

    try:
        return ZSS.index(Zs)
    except ValueError:
        raise ValueError(f"Zs ({Zs}) not found in ZSS.")
    
# ----------------------------------------------------------------------------------------------------
def get_EGMF_label(has_magnetic_field):

    if has_magnetic_field == False:
        return 'NoEGMF'

    elif has_magnetic_field == True:
        return 'EGMF'

# ----------------------------------------------------------------------------------------------------
def get_color(Zs, idist):

    if Zs == 1:
        return cm.PuRd(np.linspace(0, 1, 10)[8 - idist])
    
    elif Zs == 2:
        return cm.BuGn(np.linspace(0, 1, 10)[8 - idist])
    
    elif Zs == 7:
        return cm.OrRd(np.linspace(0, 1, 10)[8 - idist])
    
    elif Zs == 14:
        return cm.GnBu(np.linspace(0, 1, 10)[8 - idist])
    
    elif Zs == 26:
        return cm.BuPu(np.linspace(0, 1, 10)[8 - idist])

# ----------------------------------------------------------------------------------------------------
def plot_spectrum_xchecks(Zs, has_magnetic_field):

    data = np.loadtxt(f"{RESULTS_DIR}/spec_xchecks_{PARTICLES[iZs(Zs)]}_{get_EGMF_label(has_magnetic_field)}.dat")
    
    spec = np.zeros(data.shape[0])
    
    for idist in range(1, data.shape[1]):
        spec += data[:, idist]
    
    plt.plot(np.log10(data[:, 0]), spec * data[:, 0]**2, color = 'k')

    for idist in range(data.shape[1] - 1):
        plt.plot(np.log10(data[:, 0]), data[:, idist + 1] * data[:, 0]**2, color = get_color(Zs, idist))

    plt.yscale('log')
    plt.xlim([18, 21])
    plt.ylim([1e0, 1e8])
    plt.xlabel(r'$\log_{10}(\rm Energy / eV)$')
    plt.ylabel(r'$E^2 \times {\rm Intensity} \: \rm [arb. units]$')
    plt.legend([r'${\rm All}$', r'$[1, 3] \: \rm Mpc$', r'$[3, 9] \: \rm Mpc$', r'$[9, 27] \: \rm Mpc$', r'$[27, 81] \: \rm Mpc$', r'$[81, 243] \: \rm Mpc$', r'$[243, 729] \: \rm Mpc$'])
    plt.savefig(f"{FIGURES_DIR}/spectrum_xchecks_{PARTICLES[iZs(Zs)]}_{get_EGMF_label(has_magnetic_field)}.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/spectrum_xchecks_{PARTICLES[iZs(Zs)]}_{get_EGMF_label(has_magnetic_field)}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for Zs in ZSS:
        plot_spectrum_xchecks(Zs, False)
        plot_spectrum_xchecks(Zs, True)

# ----------------------------------------------------------------------------------------------------