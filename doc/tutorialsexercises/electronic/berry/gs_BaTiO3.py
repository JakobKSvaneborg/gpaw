from ase import Atoms
from ase.optimize import BFGS
from gpaw import GPAW, PW
from ase.filters import FrechetCellFilter

params = {'mode': {'name': 'pw', 'ecut': 800},
         'xc': 'PBE',
         'kpts': {'size': (3, 3, 3), 'gamma': True},
	 'convergence': {'forces': 1e-3, 'density': 1e-6},
	 'txt': 'relax.txt'}

atoms = Atoms('BaTiO3',
              cell=[3.97624117, 3.97624117, 4.29519778],
              pbc=True,
              scaled_positions=[[0.0, 0.0, 0.0292438349],
                                [0.5, 0.5, 0.546102840],
                                [0.5, 0.5, 0.965876790],
                                [0.5, 0.0, 0.488398303],
                                [0.0, 0.5, 0.488398303]])

calc = GPAW(**params)
atoms.calc = calc

uf = FrechetCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0])
opt = BFGS(uf)
opt.run(fmax=0.01)
atoms.calc.write('BaTiO3.gpw')
