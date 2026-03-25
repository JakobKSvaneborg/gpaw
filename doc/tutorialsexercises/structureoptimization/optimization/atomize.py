from ase import Atoms
from ase.parallel import paropen as open
from gpaw import GPAW

a = 10.  # Size of unit cell (Angstrom)
c = a / 2

# Hydrogen atom:
atom = Atoms('H',
             positions=[(c, c, c)],
             magmoms=[0],
             cell=(a, a + 0.0001, a + 0.0002))  # break cell symmetry

# gpaw calculator:
calc = GPAW(mode='pw',
            xc='PBE',
            hund=True,
            txt='H.out')
atom.calc = calc

e1 = atom.get_potential_energy()
calc.write('H.gpw')

# Hydrogen molecule:
d = 0.74  # experimental bond length
molecule = Atoms('H2',
                 positions=([c - d / 2, c, c],
                            [c + d / 2, c, c]),
                 cell=(a, a, a))

calc = calc.new(hund=False,  # no hund rule for molecules
                txt='H2.out')

molecule.calc = calc
e2 = molecule.get_potential_energy()
calc.write('H2.gpw')
