from matplotlib import lines
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
def select_color_for_rs(rs):

    if rs == 3:
        return cm.Blues(np.linspace(0, 1, 10)[7])
    
    elif rs == 27:
        return cm.Blues(np.linspace(0, 1, 10)[6])

    elif rs == 243:
        return cm.Blues(np.linspace(0, 1, 10)[5])

# ----------------------------------------------------------------------------------------------------
def plot_sph_harmonics_coeffs():

    x = np.logspace(-3, 3, 100)
    plt.plot(x, x, c = 'gray', ls = ':')
    plt.axhline(y = 3, c = 'gray', ls = ':')

    for rs in [27, 243]:
        data = np.loadtxt(f"{RESULTS_DIR}/sph_harmonics_coeffs_1_{rs}Mpc.dat")
        Lang2021_data = np.loadtxt(f"{REFERENCES_DIR}/Lang2021_dlt_lbd_scatt_over_rs_{rs}Mpc.dat")
        plt.plot(data[:,0], data[:,1], c = select_color_for_rs(rs))
        plt.plot(Lang2021_data[:,0], Lang2021_data[:,1], c = select_color_for_rs(rs), ls = '--')

    LAD_lgnd = lines.Line2D([], [], c = 'k', ls = '-', label = r'LAD')
    Lang2021_lgnd = lines.Line2D([], [], c = 'k', ls = '--', label = r'Lang et al. (2021)')
    plt.gca().add_artist(plt.legend(handles = [LAD_lgnd, Lang2021_lgnd], frameon = True, loc = 'lower right', bbox_to_anchor = (0.8, 0.)))

    rs3Mpc_lgnd = lines.Line2D([], [], c = select_color_for_rs(3), ls = '-', label = r'$3$')
    rs27Mpc_lgnd = lines.Line2D([], [], c = select_color_for_rs(27), ls = '-', label = r'$27$')
    rs243Mpc_lgnd = lines.Line2D([], [], c = select_color_for_rs(243), ls = '-', label = r'$243$')
    plt.gca().add_artist(plt.legend(title = r'$r_s \: \rm [Mpc]$', handles = [rs3Mpc_lgnd, rs27Mpc_lgnd, rs243Mpc_lgnd], frameon = True, loc = 'lower right'))

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([1e-3, 1e3])
    plt.ylim([1e-3, 5])
    plt.xlabel(r'$\lambda_{\rm scatt} / r_s$')
    plt.ylabel(r'Dipole amplitude')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_1.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_1.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    plot_sph_harmonics_coeffs()

# ----------------------------------------------------------------------------------------------------