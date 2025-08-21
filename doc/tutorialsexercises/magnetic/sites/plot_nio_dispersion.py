import numpy as np
import matplotlib.pyplot as plt
from gpaw.response.heisenberg import get_q0_index

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

def get_magnon_dispersion(J_qab, m_a, q0):
    # Magnon energies obtained from PRB 88 134427
    J0_ab = J_qab[q0]
    Na = len(J0_ab)
    H_qab = -J_qab
    for a in range(Na):
        H_qab[:, a] *= np.sign(m_a[a])
    H_qab[:] += np.diag(np.dot(J0_ab, np.sign(m_a)))
    H_qab *= 2 / m_a[0]
    E_qn, _ = np.linalg.eig(H_qab)
    return np.sort(E_qn, axis=1)

# Load LDA MFT data
q_qc = np.load('q_qc.npy')
q0 = get_q0_index(q_qc)
pathq_q = np.load('pathq_q.npy')
J_qab = np.load('J_qab.npy')
m_a = np.load('m_a.npy')
m_a = 2 * np.sign(m_a) # Treat as exact spin=1 system

# Plot LDA dispersion
E_qn = get_magnon_dispersion(J_qab, m_a, q0)
plt.plot(pathq_q, E_qn[:, 1].real * 1000, '-', c='C0', label='LDA')

# Load LDA+U MFT data
J_qab = np.load('J_U_qab.npy')
m_a = np.load('m_U_a.npy')
m_a = 2 * np.sign(m_a) # Treat as exact spin=1 system

# Plot LDA+U dispersion
E_qn = get_magnon_dispersion(J_qab, m_a, q0)
plt.plot(pathq_q, E_qn[:, 1].real * 1000, '-', c='C1', label='LDA+U')

# High-symmetry points
sp_p = ['L', r'$\Gamma$', 'Z', 'F', r'$\Gamma$']
spq_pc = [np.array(qc) for qc in
          [[0.5, 0, 0],
           [0, 0, 0],
           [0.5, 0.5, 0.5],
           [0.5, 0.5, 0.0],
           [0, 0, 0]]]
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

plt.yticks(size=12)
plt.ylabel(r'$\omega\;\mathrm{[meV]}$', size=18)
plt.axis([pathq_q[0], pathq_q[-1], 0, 200])
plt.legend()
plt.tight_layout()

plt.show()
