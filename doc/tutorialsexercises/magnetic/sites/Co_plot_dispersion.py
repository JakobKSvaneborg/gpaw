# web-page: Co_dispersion.png
"""Plot the magnon dispersion of Co(hcp)"""

# General modules
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

# Script modules
from ase.io import read
from gpaw.response.heisenberg import calculate_fm_magnon_energies


# ----- Load data ----- #

# Magnetic moment and J(q)
magmom = np.load('Co_magmom.npy')
q_qc = np.load('Co_q_qc.npy')
J_qab = np.load('Co_J_qab.npy')

# High-symmetry points
q_pc = np.load('Co_q_pc.npy')
sp_p = [r'$\Gamma$', 'M', 'K', 'A']

# ----- Calculate magnon energies ----- #

# Calculate dispersion and sort the modes
mm_a = np.array([magmom, magmom])
E_qn = calculate_fm_magnon_energies(J_qab, q_qc, mm_a)
E_qn = np.sort(E_qn, axis=1)

# ----- Plot the magnon dispersion ----- #

# Convert relative q-points into distance along the bandpath in reciprocal
# space
atoms = read('Co.json')
B_cv = 2.0 * np.pi * atoms.cell.reciprocal()  # Coordinate transform
q_qv = q_qc @ B_cv  # Transform into absolute reciprocal coordinates
pathq_q = [0.]
for q in range(1, len(q_qc)):
    pathq_q.append(pathq_q[-1] + np.linalg.norm(q_qv[q] - q_qv[q - 1]))
pathq_q = np.array(pathq_q)

# Define q-limits of plot
qlim = ((pathq_q[0] - pathq_q[1]) / 2.,
        (1.5 * pathq_q[-1] - 0.5 * pathq_q[-2]))

# Plot one mode at a time
colors = rcParams['axes.prop_cycle'].by_key()['color']
for n in range(2):
    plt.plot(pathq_q, E_qn[:, n] * 1e3,  # eV -> meV
             '-o', color=colors[0], mec='k')

# Use high-symmetry points as tickmarks for the x-axis
qticks = []
qticklabels = []
for sp, spq_c in zip(sp_p, q_pc):
    for q, q_c in enumerate(q_qc):
        if np.allclose(q_c, spq_c):
            qticks.append(pathq_q[q])
            qticklabels.append(sp)
plt.xticks(qticks, qticklabels)
# Plot also vertical lines for each special point
for pq in qticks:
    plt.axvline(pq, color='0.5', linewidth=1, zorder=0)

# Labels and limits
plt.xlim(qlim)
plt.ylabel(r'$\hbar\omega$ [meV]')
plt.ylim((0., 600.))

plt.savefig('Co_dispersion.png', format='png',
            bbox_inches='tight')
