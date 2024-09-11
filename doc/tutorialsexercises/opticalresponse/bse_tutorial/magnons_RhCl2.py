import numpy as np
from gpaw.response.bse import BSE

cutoff = 300

bse = BSE('gs_RhCl2.gpw',
          add_soc=True,
          ecut=cutoff,
          valence_bands=[54, 55, 56, 57],
          conduction_bands=[58, 59, 60, 61, 62, 63],
          eshift=2.4,
          nbands=40,
          truncation='2D',
          q_c=[0, 0, 0],
          txt='bse_RhCl2.txt')

w_w = np.linspace(0, 5, 5001)
modes_Gc = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]
bse.get_magnetic_susceptibility(eta=0.1,
                                modes_Gc=modes_Gc,
                                susc_component='+-',
                                write_eig='chi+-',
                                w_w=w_w)
