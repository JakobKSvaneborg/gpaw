"""Ensure that all effective volumes are the same in graphene"""
import numpy as np
from ase import Atoms

from gpaw.analyse.hirshfeld import HirshfeldPartitioning
from gpaw import GPAW, FermiDirac


c = 3
a = 1.42
atoms = Atoms(
    'C2', [(0, 0, 0.5), (1 / 3, 1 / 3, 0.5)], pbc=(1, 1, 0))
atoms.set_cell(
    [(np.sqrt(3) * a / 2.0, 3 / 2.0 * a, 0),
     (-np.sqrt(3) * a / 2.0, 3 / 2.0 * a, 0),
     (0, 0, 2 * c)],
    scale_atoms=True)
graphene = atoms.repeat([1, 2, 1])

h = 0.25
calc = GPAW(
    mode='fd',
    h=h,
    occupations=FermiDirac(0.1))
graphene.calc = calc
graphene.get_potential_energy()
vol_a = HirshfeldPartitioning(calc).get_effective_volume_ratios()
print(vol_a)
assert np.ptp(vol_a) < 1e-2
