from matplotlib.offsetbox import AnchoredText
from matplotlib.pylab import cm
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'legend.fontsize': 'x-large',
'legend.title_fontsize': 'x-large',
'axes.labelsize': 'xx-large',
'axes.titlesize': 'xx-large',
'xtick.labelsize': 'x-large',
'ytick.labelsize': 'x-large'})

FIGURES_DIR = "../figures"
REFERENCES_DIR = "../references"
RESULTS_DIR = "../results"

# ----------------------------------------------------------------------------------------------------
def get_galaxy_color(galaxy_type):

    if galaxy_type == 'RG': 
        return cm.cool(np.linspace(0, 1, 10)[8])

    elif galaxy_type == 'RG+SBG':
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
def plot_Auger_dipole_amplitude():

    Auger_01_data = np.loadtxt(f"{REFERENCES_DIR}/Auger_2020_01.dat")
    plt.errorbar(np.log10(Auger_01_data[:,0] * 1e18), Auger_01_data[:,1], [Auger_01_data[:,1] - Auger_01_data[:,2], Auger_01_data[:,3] - Auger_01_data[:,1]], c = 'k', ls = 'None', marker = '.')
    
    Auger_02_data = np.loadtxt(f"{REFERENCES_DIR}/Auger_2020_02.dat")
    x = np.log10(Auger_02_data[:,0] * 1e18)
    y = Auger_02_data[:,1]
    xlow = np.log10(Auger_02_data[:,2] * 1e18)  
    xhigh = np.log10(Auger_02_data[:,3] * 1e18) 
    yerr = Auger_02_data[:,1] - Auger_02_data[:,4]
    plt.errorbar(x, y, xerr = np.vstack([x - xlow, xhigh - x]), yerr = yerr, uplims = 1, c = 'k', ls = 'None', marker = 'None')

# ----------------------------------------------------------------------------------------------------
def plot_dipole_amplitude(L):

    for galaxy_type in ['RG', 'RG+SBG', 'SBG']:
        data = np.loadtxt(f"{RESULTS_DIR}/dipole_amplitude_full_sky_{galaxy_type}_{L}.dat")
        plt.plot(np.log10(data[:,0]), data[:,1], c = get_galaxy_color(galaxy_type))

    plot_Auger_dipole_amplitude()

    plt.gca().add_artist(AnchoredText(f'{format_luminosity_label(L)}', loc = 'upper left', frameon = False, prop = {'fontsize': 'x-large'}))
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.5))    
    
    plt.yscale('log')
    plt.xlim([18, 20])
    plt.ylim([3.e-4, 3])
    plt.xlabel(r'$\log_{10}(\rm Energy/ eV)$')
    plt.ylabel('Dipole amplitude')
    plt.legend(['RG', 'RG + SBG', 'SBG'], loc = 'lower right')
    plt.savefig(f"{FIGURES_DIR}/dipole_amplitude_full_sky_{L}.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/dipole_amplitude_full_sky_{L}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for L in ['L11', 'Lradio', 'Lgamma']:
        plot_dipole_amplitude(L)

# ----------------------------------------------------------------------------------------------------