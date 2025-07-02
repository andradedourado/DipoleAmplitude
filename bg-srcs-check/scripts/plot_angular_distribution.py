from matplotlib import lines
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
def plot_angular_distribution(): # Figure 2

    for i, lbd_scatt_over_rs in enumerate([0.01, 0.03, 0.1, 0.3, 1, 3]):

        lbd_scatt_over_rs_str = str(lbd_scatt_over_rs).rstrip('0').rstrip('.').replace('.', '_')

        data_LAD = np.loadtxt(f"{RESULTS_DIR}/angular_distr_{lbd_scatt_over_rs_str}_27Mpc.dat")
        data_Lang2021 = np.loadtxt(f"{REFERENCES_DIR}/Lang2021_dN_dcostheta_{lbd_scatt_over_rs_str}.dat")

        plt.plot(data_LAD[:,0], 2 * data_LAD[:,1], c = cm.OrRd(np.linspace(0, 1, 20)[15 - 2 * i]), label = lbd_scatt_over_rs)
        plt.plot(data_Lang2021[:,0], data_Lang2021[:,1], c = cm.OrRd(np.linspace(0, 1, 20)[15 - 2 * i]), ls = '--')

    LAD = lines.Line2D([], [], c = 'k', ls = '-', label = 'LAD')
    Lang2021 = lines.Line2D([], [], c = 'k', ls = '--', label = 'Lang+21')
    plt.gca().add_artist(plt.legend(title = 'Results', handles = [LAD, Lang2021], frameon = True, loc = 'upper left', bbox_to_anchor = (0.425, 1.)))

    plt.xlabel(r'$\cos{\theta}$')
    plt.ylabel(r'$dN/d\cos{\theta}$')
    plt.xlim([-1, 1])
    plt.ylim([0, 2])
    plt.legend(title = r'$\lambda_{\rm scatt} / r_s$', ncol = 2)
    plt.savefig(f"{FIGURES_DIR}/angular_distr.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/angular_distr.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
def plot_angular_distribution_xchecks(lbd_scatt_over_rs): 

    lbd_scatt_over_rs_str = str(lbd_scatt_over_rs).rstrip('0').rstrip('.').replace('.', '_')

    data_LAD = np.loadtxt(f"{RESULTS_DIR}/angular_distr_{lbd_scatt_over_rs_str}_27Mpc.dat")
    data_Lang2021 = np.loadtxt(f"{REFERENCES_DIR}/Lang2021_dN_dcostheta_{lbd_scatt_over_rs_str}.dat")

    plt.plot(data_LAD[:,0], 2 * data_LAD[:,1], c = 'k', label = lbd_scatt_over_rs)
    plt.plot(data_Lang2021[:,0], data_Lang2021[:,1], c = 'k', ls = '--')

    LAD = lines.Line2D([], [], c = 'k', ls = '-', label = 'LAD')
    Lang2021 = lines.Line2D([], [], c = 'k', ls = '--', label = 'Lang+21')
    plt.gca().add_artist(plt.legend(title = 'Results', handles = [LAD, Lang2021], frameon = True, loc = 'upper left'))

    at = AnchoredText(r'$\lambda_{{\rm scatt}} / r_s = {}$'.format(lbd_scatt_over_rs), loc = 'upper center', frameon = False, prop = {'fontsize': 'large'})
    plt.gca().add_artist(at)

    plt.xlabel(r'$\cos{\theta}$')
    plt.ylabel(r'$dN/d\cos{\theta}$')
    plt.xlim([-1, 1])
    plt.ylim([0, 2])
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    plot_angular_distribution()
    # plot_angular_distribution_xchecks(0.3) # [0.01, 0.03, 0.1, 0.3, 1, 3]

# ----------------------------------------------------------------------------------------------------