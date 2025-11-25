from ase import Atoms
from ase.optimize import BFGS
from gpaw import GPAW, PW
from ase.filters import FrechetCellFilter

a = Atoms('BaTiO3',
          cell=[4.00, 4.00, 4.00 * 1.054],
          pbc=True,
          scaled_positions=[[0.0, 0.0, 0.0],
                            [0.5, 0.5, 0.5 + 0.018],
                            [0.5, 0.5, 0.0],
                            [0.5, 0.0, 0.5],
                            [0.0, 0.5, 0.5]])

a.calc = GPAW(mode=PW(800),
              xc='PBE',
              kpts={'size': (3, 3, 3), 'gamma': True},
	      convergence={'forces': 1e-3, 'density': 1e-6},
              txt='relax.txt')

uf = FrechetCellFilter(a, mask=[1, 1, 1, 0, 0, 0])
opt = BFGS(uf)
opt.run(fmax=0.01)
a.calc.write('BaTiO3.gpw')
