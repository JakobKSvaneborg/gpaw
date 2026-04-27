import os
from gpaw.wannier.wannier90 import Wannier90
from gpaw import GPAW

seed = 'Cu'

calc = GPAW(seed + '.gpw', txt=None)

w90 = Wannier90(calc,
                seed=seed,
                bands=range(20),
                orbitals_ai=[[0, 1, 4, 5, 6, 7, 8]])

w90.write_input(num_iter=1000,
                dis_num_iter=500)

os.system('wannier90.x -pp ' + seed)

w90.write_projections()
w90.write_eigenvalues()
w90.write_overlaps()

os.system('wannier90.x ' + seed)
