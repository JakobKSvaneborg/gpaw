# web-page: Co_hsp_magnons_vs_rc.png
"""Plot the magnon energy as a function of the cutoff radius rc for all the
high-symmetry points of Co (hcp)"""

import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

from gpaw.response.heisenberg import calculate_fm_magnon_energies

# ----- Load data ----- #

# Magnetic moment and ideal rc
rc = np.load('Co_rc.npy')
magmom = np.load('Co_magmom.npy')

# High-symmetry points
q_pc = np.load('Co_q_pc.npy')
sp_p = [r'$\Gamma$', 'M', 'K', 'A']

# Exchange constants calculated as a function of rc
rc_r = np.load('Co_rc_r.npy')
J_pabr = np.load('Co_J_pabr.npy')

# ----- Calculate magnon energies ----- #

# Here, we keep the magnetic moment of the sites constant to see only the
# effect of J vs rc
mm_ar = magmom * np.ones(J_pabr.shape[2:], dtype=float)
E_pnr = calculate_fm_magnon_energies(J_pabr, q_pc, mm_ar)

# We separate the acoustic and optical magnon modes by sorting them
E_pnr = np.sort(E_pnr, axis=1)

# ----- Plot magnon energy vs rc ----- #

# Make a subplot for each magnon mode
fig, axes = plt.subplots(1, 2, constrained_layout=True)
colors = rcParams['axes.prop_cycle'].by_key()['color']

# Plot the magnon energies
for p, (sp, E_nr) in enumerate(zip(sp_p, E_pnr)):
    for n, E_r in enumerate(E_nr):
        if n == 0 and p == 0:
            continue  # Do not plot the acoustic mode Gamma point
        axes[n].plot(rc_r, E_r * 1e3,  # eV -> meV
                     '-x', color=colors[p], label=sp)
# Plot ideal cutoff radius
for ax in axes:
    ax.axvline(rc, color='0.5', linestyle=':')

# Labels and limits
for n, (ax, mode) in enumerate(zip(axes, ['Acoustic', 'Optical'])):
    ax.set_title(mode)
    ax.set_xlabel(r'$r_{\mathrm{c}}\: [\mathrm{\AA}]$')
    ax.set_ylabel(r'$\hbar\omega$ [meV]')
    ax.set_xlim((0.4, 1.5))
    ax.set_ylim((200., 600.))
    ax.legend()

plt.savefig('Co_hsp_magnons_vs_rc.png', format='png',
            bbox_inches='tight')
