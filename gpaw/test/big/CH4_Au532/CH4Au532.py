from ase.io import read
from ase.optimize.bfgslinesearch import BFGSLineSearch

from gpaw import GPAW

slab = read('CH4Au532.xyz')
slab.set_cell([[7.309254, 0., 0.], [4.872836, 7.509545, 0.], [0., 0., 20.]],
              scale_atoms=False)
slab.set_pbc((1, 1, 1))

calc = GPAW(mode='fd', h=0.18, kpts=(4, 4, 1))
slab.calc = calc

dyn = BFGSLineSearch(slab, trajectory='relax.traj')
dyn.run(fmax=0.02)
