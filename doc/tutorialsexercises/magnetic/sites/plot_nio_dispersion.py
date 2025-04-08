import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from ase.io import read
from gpaw.response.heisenberg import calculate_single_site_magnon_energies, get_q0_index

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
    
# Load LDA MFT data
q_qc = np.load('q_qc.npy')
pathq_q = np.load('pathq_q.npy')
J_qab = np.load('J_qab.npy')
m_a = np.load('m_a.npy')
delta_a = np.load('delta_a.npy')

q0 = get_q0_index(q_qc)
J0_ab = J_qab[q0]
Na = len(J0_ab)

# Plot LDA dispersion
H_qab = -J_qab
for a in range(Na):
    H_qab[:, a] *= np.sign(m_a[a]) 
H_qab[:] += np.diag(np.dot(J0_ab, np.sign(m_a)))
H_qab *= 2 / m_a[0]
E_qn, _ = np.linalg.eig(H_qab)
E_qn = np.sort(E_qn, axis=1)
plt.plot(pathq_q, E_qn[:, 1].real * 1000, '-', mec='k', c= 'C0', label='LDA')

# Load LDA+U MFT data
J_qab = np.load('J_U_qab.npy')
m_a = np.load('m_U_a.npy')
delta_U_a = np.load('delta_U_a.npy')

q0 = get_q0_index(q_qc)
J0_ab = J_qab[q0]
Na = len(J0_ab)

# Plot LDA+U dispersion
H_qab = -J_qab
for a in range(Na):
    H_qab[:, a] *= np.sign(m_a[a]) 
H_qab[:] += np.diag(np.dot(J0_ab, np.sign(m_a)))
H_qab *= 2 / m_a[0]
E_qn, _ = np.linalg.eig(H_qab)
E_qn = np.sort(E_qn, axis=1)
plt.plot(pathq_q, E_qn[:, 1].real * 1000, '-', mec='k', c= 'C0', label='LDA+U')


# High-symmetry points
sp_p = ['L', r'$\Gamma$', 'Z', 'F', r'$\Gamma$']
spq_pc = [np.array(qc) for qc in
          [[0.5, 0, 0.0], [0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0, 0, 0]]]

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
plt.ylabel(r'$\omega\;\mathrm{[eV]}$', size=16)
plt.axis([pathq_q[0], pathq_q[-1], 0, 400])
plt.legend()
plt.tight_layout()

plt.show()
