from ase import Atoms
from ase.optimize import BFGS
from gpaw import GPAW
from ase.filters import FrechetCellFilter

params = {'mode': {'name': 'pw', 'ecut': 800},
          'xc': 'PBE',
          'kpts': {'size': (8, 8, 8),
                   'gamma': True},
          'convergence': {'forces': 1e-3,
                          'density': 1e-6},
          'txt': 'relax.txt'}

atoms = Atoms('BaTiO3',
              cell=[3.976241, 3.976241, 4.295198],
              pbc=True,
              scaled_positions=[[0.0, 0.0, 0.029],
                                [0.5, 0.5, 0.546],
                                [0.5, 0.5, 0.966],
                                [0.5, 0.0, 0.488],
                                [0.0, 0.5, 0.488]])

calc = GPAW(**params)
atoms.calc = calc

uf = FrechetCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0])
opt = BFGS(uf)
opt.run(fmax=0.01)
atoms.calc.write('BaTiO3.gpw')
