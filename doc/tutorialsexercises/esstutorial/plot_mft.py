import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from gpaw.response.heisenberg import get_q0_index

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
S = 5 / 2

# Load MFT data
q_qc = np.load('q_qc_path.npy')
J_qab = np.load('J_60_qab_path.npy')
J_qab /= S**2
m_a = [5, -5]

q0 = get_q0_index(q_qc)
J0_ab = J_qab[q0]
Na = len(J0_ab)

# Convert relative q-points into distance along the bandpath
atoms = read('gs_afm.gpw')
B_cv = 2.0 * np.pi * atoms.cell.reciprocal()
q_qv = q_qc @ B_cv
pathq_q = [0.]
for q in range(1, len(q_qc)):
    pathq_q.append(pathq_q[-1] + np.linalg.norm(q_qv[q] - q_qv[q - 1]))
pathq_q = np.array(pathq_q)

# Calculate the magnon energies without anisotropy
H_qab = np.zeros((len(q_qc), Na, Na), complex)
for iq in range(len(q_qc)):
    for a in range(Na):
        for b in range(Na):
            H_qab[iq, a, b] -= np.sign(m_a[a]) * J_qab[iq, a, b]
            if a == b:
                for c in range(Na):
                    H_qab[iq, a, a] += np.sign(m_a[c]) * J0_ab[a, c]
H_qab *= S
E_qn, v_qan = np.linalg.eig(H_qab)
E_qn *= 1000
E_qn = np.sort(E_qn, axis=1)

# Plot magnon dispersion
plt.plot(pathq_q, -E_qn[:, 0].real, '-o', c='C0', lw=2, ms=5)
plt.plot(pathq_q, E_qn[:, 1].real, '-o', c='C1', lw=2, ms=5)

# High-symmetry points
sp_p = ['X', r'$\Gamma$', 'M', 'A', 'Z', 'R']
spq_pc = [np.array(qc) for qc in
          [[0.5, 0, 0], [0, 0, 0], [0.5, 0.5, 0],
           [0.5, 0.5, 0.5], [0, 0, 0.5], [0.5, 0, 0.5]]]
qticks = []
qticklabels = []
for sp, spq_c in zip(sp_p, spq_pc):
    for q, q_c in enumerate(q_qc):
        if np.allclose(q_c, spq_c):
            qticks.append(pathq_q[q])
            qticklabels.append(sp)
plt.xticks(qticks, qticklabels, size=16)
for pq in qticks:
    plt.axvline(pq, color='0.5', linewidth=1, zorder=0)

plt.yticks(size=10)
plt.ylabel(r'$\omega\;\mathrm{[meV]}$', size=16)
plt.axis([pathq_q[0], pathq_q[-1], 0, None])
plt.tight_layout()

plt.show()
