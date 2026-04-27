import numpy as np
from ase.io import read

# Load MFT data
S = 5 / 2
q_qc = np.load('q_qc.npy')
J_qab = np.load('J_n60_qab.npy')
J_qab *= 1000 / S**2
Na = len(J_qab[0])
Nq = len(J_qab)

Mn2 = read('gs_afm.gpw')[:2]
pos_av = Mn2.get_positions()

R_i = []
J_i = []
for R_c in [[0, 0, 0], [0, 0, 1], [1, 0, 0],
            [0, 1, 0], [1, 1, 0], [1, -1, 0]]:
    exp_q = np.exp(2 * np.pi * 1.0j * np.dot(q_qc, R_c))
    J_ab = np.dot(exp_q, np.swapaxes(J_qab, 0, 1)) / Nq
    if R_c == [0, 0, 0]:
        J_ab[0, 0] = 0
        J_ab[1, 1] = 0
    R_v = np.dot(Mn2.get_cell().T, R_c)
    R_ab = np.array([[np.linalg.norm(R_v),
                      np.linalg.norm(R_v + pos_av[0] - pos_av[1])],
                     [np.linalg.norm(R_v + pos_av[1] - pos_av[0]),
                      np.linalg.norm(R_v)]])
    print('R_c: ', R_c)
    print('R_ab:')
    print(np.round(R_ab, 2))
    R_i.append(R_ab.flatten())
    print('J_ab:')
    print(np.round(J_ab.real, 5))
    J_i.append(J_ab.real.flatten())
    print()

R_i = np.array(R_i).flatten()
J_i = np.array(J_i).flatten()

argsort = np.argsort(R_i)

R_i = R_i[argsort]
J_i = J_i[argsort]
