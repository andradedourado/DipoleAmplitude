from matplotlib.offsetbox import AnchoredText
from matplotlib.pylab import cm
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'legend.fontsize': 'large',
'legend.title_fontsize': 'large',
'axes.labelsize': 'x-large',
'axes.titlesize': 'xx-large',
'xtick.labelsize': 'x-large',
'ytick.labelsize': 'x-large'})

FIGURES_DIR = "../figures"
REFERENCES_DIR = "../references"
RESULTS_DIR = "../results"

# ----------------------------------------------------------------------------------------------------
def get_galaxy_color(galaxy_type):

    if galaxy_type == 'AGN': 
        return cm.cool(np.linspace(0, 1, 10)[8])

    elif galaxy_type == 'AGN+SBG':
        return cm.cool(np.linspace(0, 1, 10)[5])

    elif galaxy_type == 'SBG':
        return cm.cool(np.linspace(0, 1, 10)[2])

# ----------------------------------------------------------------------------------------------------
def format_luminosity_label(L):

    if L == 'L11':
        return r'$L_{\rm 1:1}$'

    elif L == 'Lradio':
        return r'$L_{\rm radio}$'

    elif L == 'Lgamma':
        return r'$L_\gamma$'

# ----------------------------------------------------------------------------------------------------
def plot_dipole_amplitude(L):

    for galaxy_type in ['AGN', 'AGN+SBG', 'SBG']:
        data = np.loadtxt(f"{RESULTS_DIR}/dipole_amplitude_full_sky_{galaxy_type}_{L}.dat")
        plt.plot(np.log10(data[:,0]), data[:,1], c = get_galaxy_color(galaxy_type))

    Auger_01_data = np.loadtxt(f"{REFERENCES_DIR}/Auger_2020_01.dat")
    plt.errorbar(np.log10(Auger_01_data[:,0] * 1e18), Auger_01_data[:,1], [Auger_01_data[:,1] - Auger_01_data[:,2], Auger_01_data[:,3] - Auger_01_data[:,1]], c = 'k', ls = 'None', marker = '.')

    plt.gca().add_artist(AnchoredText(f'{format_luminosity_label(L)}', loc = 'upper left', frameon = False, prop = {'fontsize': 'x-large'}))

    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}' if x.is_integer() else f'{x}'))
    
    plt.yscale('log')
    plt.xlabel(r'$\log_{10}(\rm Energy/ eV)$')
    plt.ylabel('Dipole amplitude')
    plt.legend(['AGN', 'AGN + SBG', 'SBG'], loc = 'lower right')
    plt.savefig(f"{FIGURES_DIR}/dipole_amplitude_full_sky_{L}.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/dipole_amplitude_full_sky_{L}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for L in ['L11', 'Lradio', 'Lgamma']:
        plot_dipole_amplitude(L)

# ----------------------------------------------------------------------------------------------------