from gpaw.response.bse import BSE
from gpaw import GPAW
import numpy as np
from gpaw.mpi import world

ecut=50
bse = BSE('scf_PW.gpw',
          ecut=ecut,
          valence_bands = 6, #range(6, 9),
          conduction_bands = 6, #range(9, 10),
          nbands=100,
          mode='BSE',
          txt='bse_Si.txt')

chi = bse.get_chi_wGG(w_w=np.linspace(0, 50, 5001), eta=0.1, optical=True,
                      write_eigenstates=True)

if world.rank == 0:
    np.save('chi.npy', chi)


chi = bse.get_chi_wGG(w_w=np.linspace(0, 50, 5001), eta=0.1, optical=True,
                      write_eigenstates=False, read_eigenstates=True)

if world.rank == 0:
    np.save('chi_from_saved_eigs.npy', chi)

