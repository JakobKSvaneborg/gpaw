# creates: h2.txt
from ase import Atoms
from gpaw import GPAW

d = 0.74
a = 6.0

atoms = Atoms('H2',
              positions=[(0, 0, 0),
                         (0, 0, d)],
              cell=(a, a, a))
atoms.center()

calc = GPAW(mode='pw', txt='h2.txt')
atoms.calc = calc
print(atoms.get_forces())
