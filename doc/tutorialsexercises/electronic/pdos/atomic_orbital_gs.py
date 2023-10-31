from ase.build import bulk
from gpaw import GPAW

atoms = bulk('Au')
k = 8
atoms.calc = GPAW(mode='pw',
                  kpts=(k, k, k))
atoms.get_potential_energy()
atoms.calc.write('au.gpw')
