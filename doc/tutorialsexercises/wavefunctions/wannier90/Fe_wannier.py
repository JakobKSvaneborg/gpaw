import os
from gpaw.wannier.wannier90 import Wannier90
from gpaw import GPAW

seed = 'Fe'

calc = GPAW('Fe.gpw')

w90 = Wannier90(calc,
                seed=seed,
                bands=range(30),
                spinors=True)

w90.write_input(num_iter=200,
                dis_num_iter=500,
                dis_mix_ratio=1.0,
                dis_froz_max=15.0)

os.system('wannier90.x -pp ' + seed)

w90.write_projections()
w90.write_eigenvalues()
w90.write_overlaps()

os.system('wannier90.x ' + seed)
