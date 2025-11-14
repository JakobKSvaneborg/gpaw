from math import sqrt

import numpy as np
from ase import Atoms
from ase.build import hcp0001

from gpaw import GPAW

# Vacuum and hcp lattice parameter for graphene
d = 4.0
a = 2.4437

# Calculate potential energy per atom for orthogonal unitcell
atoms = hcp0001('C', a=a / sqrt(3), vacuum=d, size=(3, 2, 1), orthogonal=True)
del atoms[[1, -1]]
atoms.center(axis=0)
kpts_c = np.ceil(50 / np.sum(atoms.get_cell()**2, axis=1)**0.5).astype(int)
kpts_c[~atoms.get_pbc()] = 1
calc = GPAW(mode='fd', h=0.15, xc='LDA', nbands=-4, kpts=kpts_c, txt='-',
            basis='dzp', convergence={'energy': 1e-5, 'density': 1e-5})
atoms.calc = calc
eppa1 = atoms.get_potential_energy() / len(atoms)
F1_av = atoms.get_forces()
assert np.abs(F1_av).max() < 5e-3

# Redo calculation with non-orthogonal unitcell
atoms = Atoms(symbols='C2', pbc=(True, True, False),
              positions=[(a / 2, -sqrt(3) / 6 * a, d),
                         (a / 2, sqrt(3) / 6 * a, d)],
              cell=[(a / 2, -sqrt(3) / 2 * a, 0),
                    (a / 2, sqrt(3) / 2 * a, 0),
                    (0, 0, 2 * d)])
kpts_c = np.ceil(50 / np.sum(atoms.get_cell()**2, axis=1)**0.5).astype(int)
kpts_c[~atoms.get_pbc()] = 1
atoms.calc = calc.new(kpts=kpts_c)
eppa2 = atoms.get_potential_energy() / len(atoms)
F2_av = atoms.get_forces()
assert np.abs(F2_av).max() < 5e-3
assert abs(eppa1 - eppa2) < 1e-3
