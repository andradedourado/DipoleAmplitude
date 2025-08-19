import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "../figures"
RESULTS_DIR = "../results"

PARTICLES = ['1H', '4He', '14N', '28Si', '56Fe']
PARTICLES_LEGEND = [r'$^1\mathrm{H}$', r'$^4\mathrm{He}$', r'$^{14}\mathrm{N}$', r'$^{28}\mathrm{Si}$', r'$^{56}\mathrm{Fe}$']
ZS = [1, 2, 7, 14, 26]

plt.rcParams.update({'legend.fontsize': 'large',
'legend.title_fontsize': 'large',
'axes.labelsize': 'x-large',
'axes.titlesize': 'x-large',
'xtick.labelsize': 'x-large',
'ytick.labelsize': 'x-large'})

# ----------------------------------------------------------------------------------------------------
def iZ(Z):

    try:
        return ZS.index(Z)
    except ValueError:
        raise ValueError(f"Z ({Z}) not found in ZS.")

# ----------------------------------------------------------------------------------------------------
def plot_density_maps(Z, galaxy, EGMF, nside):

    density_map = np.loadtxt(f"{RESULTS_DIR}/density_map_{PARTICLES[iZ(Z)]}_{galaxy}_EGMF{EGMF:02d}_{nside:02d}.dat")
    density_map_norm = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map))

    hp.projview(density_map_norm, cmap = 'coolwarm', graticule = True, graticule_labels = True, projection_type = 'mollweide', 
                override_plot_properties = {'cbar_pad': 0.0875})
    
    plt.xlabel(r'Galactic longitude, $l \: {\rm [deg]}$', labelpad = 8.75)
    plt.ylabel(r'Galactic latitude, $b \: {\rm [deg]}$')
    plt.title(r'{} | Centaurus A | Seed {}'.format(PARTICLES_LEGEND[iZ(Z)], EGMF), pad = 8.75)
    plt.grid(linestyle = 'dotted', color = 'black', zorder = -1.0)
    plt.savefig(f'{FIGURES_DIR}/density_map_{PARTICLES[iZ(Z)]}_{galaxy}_EGMF{EGMF:02d}_{nside:02d}.pdf', bbox_inches = 'tight')
    plt.savefig(f'{FIGURES_DIR}/density_map_{PARTICLES[iZ(Z)]}_{galaxy}_EGMF{EGMF:02d}_{nside:02d}.png', bbox_inches = 'tight', dpi = 300)
    plt.show()

# ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    for EGMF in range(20):
        plot_density_maps(1, 'CenA', EGMF, 64)

# ----------------------------------------------------------------------------------------------------