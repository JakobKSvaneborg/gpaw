import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.directmin.etdm_fdpw import FDPWETDM

# Water molecule:
d = 0.9575
t = np.pi / 180 * 104.51
H2O = Atoms('OH2',
            positions=[(0, 0, 0),
                       (d, 0, 0),
                       (d * np.cos(t), d * np.sin(t), 0)])
H2O.center(vacuum=5.0)

calc = GPAW(mode='pw',
            eigensolver=FDPWETDM(converge_unocc=False),
            mixer={'backend': 'no-mixing'},
            occupations={'name': 'fixed-uniform'},
            spinpol=True)
H2O.set_calculator(calc)
H2O.get_potential_energy()
