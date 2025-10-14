from matplotlib import lines
from matplotlib.pylab import cm
from scipy.special import erf
from scipy.special import k1
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'legend.fontsize': 'x-large',
'legend.title_fontsize': 'x-large',
'axes.labelsize': 'xx-large',
'axes.titlesize': 'xx-large',
'xtick.labelsize': 'x-large',
'ytick.labelsize': 'x-large'})

FIGURES_DIR = "../figures"

# ----------------------------------------------------------------------------------------------------
def diffusive_distr(cts, lbd_scatt, r):

    sgm = np.sqrt(lbd_scatt * cts / 3)
    A = (sgm**2 * (np.sqrt(np.pi / 2) * sgm * erf(cts / (2*sgm))) - cts * np.exp(-cts**2 / (2 * sgm**2)))**-1

    mask = r <= cts
    dN_dr = np.zeros_like(r)
    dN_dr[mask] = A * r[mask]**2 * np.exp(-r[mask]**2 / (2 * sgm**2))

    return dN_dr

# ----------------------------------------------------------------------------------------------------
def transition_distr(cts, lbd_scatt, r):

    alp = 3 * cts / lbd_scatt

    mask = r <= cts
    dN_dr = np.zeros_like(r)
    dN_dr[mask] = r[mask]**2 * alp * np.exp(-alp / np.sqrt(1 - (r[mask] / cts)**2)) / (cts**3 * k1(alp) * (1 - (r[mask] / cts)**2)**2)

    return dN_dr

# ----------------------------------------------------------------------------------------------------
def plot_dist_distr(lbd_scatt = 10):

    ctss = np.logspace(-1, 4, num = 11)
    r = np.logspace(-1, 4, num = 10000)

    iOr = 8
    iBl = 6
    iGr = 5

    for cts in ctss:
        
        alp = 3 * cts / lbd_scatt

        if alp < 0.1:
            plt.axvline(x = cts, c = cm.Oranges(np.linspace(0, 1, 10))[iOr])
            iOr += 1
        
        elif 0.1 <= alp <= 10:
            plt.plot(r, r * transition_distr(cts, lbd_scatt, r), c = cm.Blues(np.linspace(0, 1, 10))[iBl])
            iBl += 1
        
        elif alp > 10:
            plt.plot(r, r * diffusive_distr(cts, lbd_scatt, r), c = cm.Greens(np.linspace(0, 1, 10))[iGr])
            iGr += 1

    ballistic = lines.Line2D([], [], c = cm.Oranges(np.linspace(0, 1, 10))[8], ls = '-', label = r'$\alpha < 0.1$')
    transition = lines.Line2D([], [], c = cm.Blues(np.linspace(0, 1, 10))[6], ls = '-', label = r'$0.1 \leq \alpha \leq 10$')
    diffusive = lines.Line2D([], [], c = cm.Greens(np.linspace(0, 1, 10))[5], ls = '-', label = r'$\alpha > 10$')
    plt.gca().add_artist(plt.legend(title = r'$\alpha = 3 ct_s / \lambda_{\rm scatt}$', handles = [ballistic, transition, diffusive], frameon = True, loc = 'upper right'))

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-2, 1e2])
    plt.xlabel(r'$r \: [\rm Mpc]$')
    plt.ylabel(r'$(r/c)[d^2N/(drdt)] \: [\rm Mpc^{-1}]$')
    plt.title(r'$\lambda_{\rm scatt} = 10 \: {\rm Mpc} \: | \: ct_{max} = 10^4 \: {\rm Mpc}$')
    plt.savefig(f"{FIGURES_DIR}/dist_distr.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/dist_distr.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    plot_dist_distr()

# ----------------------------------------------------------------------------------------------------