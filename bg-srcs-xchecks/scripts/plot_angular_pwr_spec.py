from matplotlib.offsetbox import AnchoredText
from matplotlib.pylab import cm
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
def plot_angular_power_spectrum():

    for l in range(5):
        LAD_data = np.loadtxt(f"{RESULTS_DIR}/angular_pwr_spec_27Mpc_{l:02d}.dat")
        Lang2021_data = np.loadtxt(f"{REFERENCES_DIR}/Lang2021_Cl_lbd_scatt_over_rs_{l:02d}.dat")
        plt.plot(LAD_data[:,0], np.sqrt(LAD_data[:,1]), c = cm.Greens(np.linspace(0, 1, 20)[19-3*l]), label = r'$l = {}$'.format(l))
        plt.plot(Lang2021_data[:,0], Lang2021_data[:,1], c = cm.Greens(np.linspace(0, 1, 20)[19-3*l]), ls = '--', label = '_nolegend_')

    # Lang2021_data = np.loadtxt(f"{REFERENCES_DIR}/Lang2021_Cl_lbd_scatt_over_rs_f_non-dip.dat")
    # plt.plot(Lang2021_data[:,0], Lang2021_data[:,1], c = 'k', ls = '--', label = r'$f_{\rm non-dip}$')

    at = AnchoredText(r'$r_s = 27 \: \rm Mpc$', loc = 'lower right', frameon = False, prop = {'fontsize': 'large'})
    plt.gca().add_artist(at)

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(9e-4, 2e0)
    plt.xlabel(r'$\lambda_{\rm scatt}/r_s$')
    plt.ylabel(r'$\sqrt{C_l}$')
    plt.legend()
    plt.savefig(f"{FIGURES_DIR}/angular_power_spectrum.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/angular_power_spectrum.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    plot_angular_power_spectrum()

# ----------------------------------------------------------------------------------------------------