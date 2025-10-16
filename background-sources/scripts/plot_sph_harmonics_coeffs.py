from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.pylab import cm
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

PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
PARTICLES_LEGEND = [r'$^1\mathrm{H}$', r'$^4\mathrm{He}$', r'$^{14}\mathrm{N}$', r'$^{28}\mathrm{Si}$', r'$^{56}\mathrm{Fe}$']
ZS = [1, 2, 7, 14, 26]

# ----------------------------------------------------------------------------------------------------
def iZ(Z):

    try:
        return ZS.index(Z)
    except ValueError:
        raise ValueError(f"Z ({Z}) not found in ZS.")
    
# ----------------------------------------------------------------------------------------------------
def get_color(i, Z):

    if Z == 1:
        return cm.PuRd(np.linspace(0, 1, 10)[9 - i])
    
    elif Z == 2:
        return cm.BuGn(np.linspace(0, 1, 10)[9 - i])
    
    elif Z == 7:
        return cm.OrRd(np.linspace(0, 1, 10)[9 - i])
    
    elif Z == 14:
        return cm.GnBu(np.linspace(0, 1, 10)[9 - i])
    
    elif Z == 26:
        return cm.BuPu(np.linspace(0, 1, 10)[9 - i])
    
# ----------------------------------------------------------------------------------------------------
def number_density(Dshell):

    return 1e4 / (4 * np.pi * Dshell**3) # 1e-4 Mpc^-3

# ----------------------------------------------------------------------------------------------------
def plot_sph_harmonics_coeffs(Dshell):

    for Z in ZS:
        
        Phi_0_tot = np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/Phi_0_tot_{PARTICLES[iZ(Z)]}.dat")
        Phi_1_tot = np.loadtxt(f"{RESULTS_DIR}/{int(Dshell)}Mpc/Phi_1_tot_{PARTICLES[iZ(Z)]}.dat")

        dlt = Phi_1_tot[:,1] / Phi_0_tot[:,1]
        mask = (dlt > 0) & np.isfinite(dlt) 

        plt.plot(np.log10(Phi_0_tot[:,0][mask]), dlt[mask], c = get_color(2, Z), label = PARTICLES_LEGEND[iZ(Z)])

    plt.gca().add_artist(AnchoredText(r'$n = {:.2f} \times 10^{{-4}} \: \rm Mpc^{{-3}}$'.format(number_density(Dshell)), loc = 'upper left', frameon = False, prop = {'fontsize': 'x-large'}))
    plt.yscale("log")
    plt.xlim([18, 20])
    plt.ylim([1e-5, 3])
    plt.xlabel(r"$\log_{10}{(\rm Energy / eV)}$")
    plt.ylabel(r"Dipole amplitude")
    plt.legend(title = 'Nucleus', loc = 'lower right')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_{int(Dshell)}Mpc.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_{int(Dshell)}Mpc.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
   
   for Dshell in range(1, 10):
        plot_sph_harmonics_coeffs(Dshell)

# ----------------------------------------------------------------------------------------------------