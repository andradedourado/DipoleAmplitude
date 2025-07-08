from matplotlib import lines
from matplotlib.pylab import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
def plot_dilution_factor():

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:g}'))

    for iDmin, Dmin in enumerate([3, 9, 27, 81, 243]):
        LAD_data = np.loadtxt(f"{RESULTS_DIR}/spec_1H_EGMF_Dmin_{int(Dmin)}Mpc.dat")
        Lang2020_data = np.loadtxt(f"{REFERENCES_DIR}/Lang2020_dilution_factor_{int(Dmin)}Mpc.dat")
        plt.plot(np.log10(LAD_data[:,0]), LAD_data[:,1] / LAD_data[:, 1:].sum(axis = 1), c = cm.PuRd(np.linspace(0, 1, 10)[9 - 2 * iDmin]), label = r'${} \: \rm Mpc$'.format(int(Dmin)))
        plt.plot(Lang2020_data[:,0], Lang2020_data[:,1], c = cm.PuRd(np.linspace(0, 1, 10)[9 - 2 * iDmin]), ls = '--', label = '_nolabel_')

    LAD = lines.Line2D([], [], c = 'k', ls = '-', label = 'LAD')
    Lang2021 = lines.Line2D([], [], c = 'k', ls = '--', label = 'Lang+20')
    plt.gca().add_artist(plt.legend(title = 'Results', handles = [LAD, Lang2021], frameon = True, loc = 'upper left'))

    plt.yscale('log')
    plt.xlim([18,20])
    plt.xlabel(r'$\log_{10}(\rm Energy/eV)$')
    plt.ylabel('Dilution factor')
    plt.legend(title = r'$D_{\rm min}$')
    plt.savefig(f"{FIGURES_DIR}/dilution_factor.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/dilution_factor.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    plot_dilution_factor()

# ----------------------------------------------------------------------------------------------------
