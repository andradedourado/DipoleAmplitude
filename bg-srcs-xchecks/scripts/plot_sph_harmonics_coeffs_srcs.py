from matplotlib import lines
from matplotlib import pyplot as plt
from matplotlib.pylab import cm
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
def format_filename_value(B):

    B = float(B)

    if B.is_integer():
        return str(int(B))
    else:
        return str(B).replace('.', '_')
    
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
def plot_sph_harmonics_coeffs_Fig7():

    for Z in ZS:
        Phi_0_tot = np.loadtxt(f"{RESULTS_DIR}/Figures6to8/Phi_0_tot_27Mpc_{PARTICLES[iZ(Z)]}_1nG.dat")
        Phi_1_tot = np.loadtxt(f"{RESULTS_DIR}/Figures6to8/Phi_1_tot_27Mpc_{PARTICLES[iZ(Z)]}_1nG.dat")
        plt.plot(np.log10(Phi_0_tot[:,0]), Phi_1_tot[:,1] / Phi_0_tot[:,1], c = get_color(2, Z), label = PARTICLES_LEGEND[iZ(Z)])

        Lang2021_data = np.loadtxt(f"{REFERENCES_DIR}/Lang2021_dlt_log10E_27Mpc_{PARTICLES[iZ(Z)]}_1nG.dat")
        plt.plot(Lang2021_data[:,0], Lang2021_data[:,1], c = get_color(2, Z), ls = '--', label  = '__nolegend__')

    LAD_lgnd = lines.Line2D([], [], c = 'k', ls = '-', label = r'LAD')
    Lang2021_lgnd = lines.Line2D([], [], c = 'k', ls = '--', label = r'Lang et al. (2021)')
    plt.gca().add_artist(plt.legend(handles = [LAD_lgnd, Lang2021_lgnd], frameon = True, loc = 'upper left', bbox_to_anchor = (0.208, 1.)))

    plt.yscale("log")
    plt.xlabel(r"$\log_{10}{(\rm Energy / eV)}$")
    plt.ylabel(r"Dipole amplitude")
    plt.legend(title = 'Nucleus')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_Fig7.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_Fig7.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
def plot_sph_harmonics_coeffs_Fig8():

    for iB, B in enumerate([0.1, 0.3, 1, 3]):
        Phi_0_tot = np.loadtxt(f"{RESULTS_DIR}/Figures6to8/Phi_0_tot_3Mpc_14N_{format_filename_value(B)}nG.dat")
        Phi_1_tot = np.loadtxt(f"{RESULTS_DIR}/Figures6to8/Phi_1_tot_3Mpc_14N_{format_filename_value(B)}nG.dat")
        plt.plot(np.log10(Phi_0_tot[:,0]), Phi_1_tot[:,1] / Phi_0_tot[:,1], c = get_color(2*iB, 7), label = f'{B}')

        Lang2021_data = np.loadtxt(f"{REFERENCES_DIR}/Lang2021_dlt_log10E_3Mpc_14N_{format_filename_value(B)}nG.dat")
        plt.plot(Lang2021_data[:,0], Lang2021_data[:,1], c = get_color(2*iB, 7), ls = '--', label  = '__nolegend__')

    LAD_lgnd = lines.Line2D([], [], c = 'k', ls = '-', label = r'LAD')
    Lang2021_lgnd = lines.Line2D([], [], c = 'k', ls = '--', label = r'Lang et al. (2021)')
    plt.gca().add_artist(plt.legend(handles = [LAD_lgnd, Lang2021_lgnd], frameon = True, loc = 'upper left', bbox_to_anchor = (0.28, 1.)))

    plt.yscale("log")
    plt.xlabel(r"$\log_{10}{(\rm Energy / eV)}$")
    plt.ylabel(r"Dipole amplitude")
    plt.legend(title = 'Magnetic field')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_Fig8.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_Fig8.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
def plot_sph_harmonics_coeffs_Fig9(Z):

    for iDmin, Dmin in enumerate([3, 9, 27, 81, 243]):
        Phi_0_tot = np.loadtxt(f"{RESULTS_DIR}/Figures6to8/Phi_0_tot_{int(Dmin)}Mpc_{PARTICLES[iZ(Z)]}_1nG.dat")
        Phi_1_tot = np.loadtxt(f"{RESULTS_DIR}/Figures6to8/Phi_1_tot_{int(Dmin)}Mpc_{PARTICLES[iZ(Z)]}_1nG.dat")
        plt.plot(np.log10(Phi_0_tot[:,0]), Phi_1_tot[:,1] / Phi_0_tot[:,1], c = get_color(iDmin, Z), label = f'{int(Dmin)}')

        Lang2021_data = np.loadtxt(f"{REFERENCES_DIR}/Lang2021_dlt_log10E_{int(Dmin)}Mpc_{PARTICLES[iZ(Z)]}_1nG.dat")
        plt.plot(Lang2021_data[:,0], Lang2021_data[:,1], c = get_color(iDmin, Z), ls = '--', label  = '__nolegend__')

    LAD_lgnd = lines.Line2D([], [], c = 'k', ls = '-', label = r'LAD')
    Lang2021_lgnd = lines.Line2D([], [], c = 'k', ls = '--', label = r'Lang et al. (2021)')
    plt.gca().add_artist(plt.legend(handles = [LAD_lgnd, Lang2021_lgnd], frameon = True, loc = 'upper left', bbox_to_anchor = (0.215, 1.)))

    plt.yscale("log")
    plt.xlabel(r"$\log_{10}{(\rm Energy / eV)}$")
    plt.ylabel(r"Dipole amplitude")
    plt.legend(title = r'$D_{\rm min} \: \rm [Mpc]$', loc = 'upper left')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_{PARTICLES[iZ(Z)]}_Fig9.pdf", bbox_inches = 'tight')
    plt.savefig(f"{FIGURES_DIR}/sph_harmonics_coeffs_{PARTICLES[iZ(Z)]}_Fig9.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    # plot_sph_harmonics_coeffs_Fig7()

    plot_sph_harmonics_coeffs_Fig8()
    
    # for Z in ZS:
    #     plot_sph_harmonics_coeffs_Fig9(Z)

# ----------------------------------------------------------------------------------------------------