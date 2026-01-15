"""NiO in an antiferromagnetic configuration.

See:
    Quasiparticle energy bands of NiO in the GW approximation
    Je-Luen Li, G.-M. Rignanese, and Steven G. Louie
    10.1103/PhysRevB.71.193102
"""
from ase import Atoms

from gpaw import GPAW

a = 4.1767  # lattice constants
b = a / 2
m = 2.0

atoms = Atoms('Ni2O2',
              pbc=True,
              cell=[(b, b, 0),
                    (0, b, b),
                    (a, 0, a)],
              positions=[(0, 0, 0),
                         (b, 0, b),
                         (b, 0, 0),
                         (a, 0, b)],
              magmoms=[m, -m, 0, 0])
atoms.write('nio.json')

atoms.calc = GPAW(mode={'name': 'pw', 'ecut': 600},
                  txt='nio.txt',
                  kpts={'density': 3.0, 'gamma': True},
                  xc='PBE')

e = atoms.get_potential_energy()

calc = atoms.calc.fixed_density(nbands=100)
calc.write('nio.gpw', mode='all')
