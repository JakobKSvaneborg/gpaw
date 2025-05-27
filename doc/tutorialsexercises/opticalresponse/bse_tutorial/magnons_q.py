import numpy as np
from gpaw.response.bse import BSE
from gpaw.response.bse import read_bse_eigenvalues
from ase.parallel import paropen

cutoff = 300

qs_qc = [[0.5, 0, 0],
         [0.375, 0, 0],
         [0.25, 0, 0],
         [0.125, 0, 0],
         [0, 0, 0],
         [0, 0.25, 0],
         [0, 0.5, 0]]

for i, q_c in enumerate(qs_qc):
    bse = BSE('gs_RhCl2.gpw',
              add_soc=True,
              ecut=cutoff,
              valence_bands=[54, 55, 56, 57],
              conduction_bands=[58, 59, 60, 61, 62, 63],
              eshift=2.4,
              nbands=40,
              truncation='2D',
              q_c=q_c,
              txt=f'bse_RhCl2_q{i}.txt')

    bse.get_magnetic_susceptibility(eta=0.1,
                                    write_eig=f'eig_q{i}_',
                                    susc_component='+-',
                                    w_w=np.linspace(0, 1, 100))

    fd = paropen('magnons_q.dat', 'a')
    if i < 4:
        q = -2 * np.pi * np.dot(q_c, q_c)**0.5 / 3.5006
    else:
        q = 2 * np.pi * np.dot(q_c, q_c)**0.5 / 6.8529
    w_T, C_T = read_bse_eigenvalues(f'eig_q{i}_000.dat')
    print(q, w_T[0], w_T[1], file=fd)
    fd.close()
