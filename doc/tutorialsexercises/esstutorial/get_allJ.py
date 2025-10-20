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
for i in range(-2, 3):
    for j in range(-2, 3):
        for h in range(-2, 3):
            R_c = [i, j, h]
            exp_q = np.exp(2 * np.pi * 1.0j * np.dot(q_qc, R_c))
            J_ab = np.dot(exp_q, np.swapaxes(J_qab, 0, 1)) / Nq
            R_v = np.dot(Mn2.get_cell().T, R_c)
            R_ab = np.array([[np.linalg.norm(R_v),
                              np.linalg.norm(R_v + pos_av[0] - pos_av[1])],
                             [np.linalg.norm(R_v + pos_av[1] - pos_av[0]),
                              np.linalg.norm(R_v)]])
            R_i.append(R_ab.flatten())
            J_i.append(J_ab.real.flatten())

R_i = np.array(R_i).flatten()
J_i = np.array(J_i).flatten()

argsort = np.argsort(R_i)
R_i = R_i[argsort]
J_i = J_i[argsort]

Rs = []
Js = []
counter = 0
print('   d [Å]  J [meV]')
for i in range(2, len(R_i)):
    R = R_i[i]
    J = J_i[i]
    if (np.round(J, 6) in Js and R in Rs) or R > 10:
        continue
    else:
        if R not in Rs:
            counter += 1
            Rs.append(R)
        print('J%s' % counter, f'{R:2.2f}', f'{J:1.6f}')
        Js.append(np.round(J, 6))
