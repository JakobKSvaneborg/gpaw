from gpaw import GPAW
from ase.build import molecule
import numpy as np

atoms = molecule('C60')
atoms.center(vacuum=2)
calc = GPAW(mode={'name': 'pw',
                  'ecut': 200,
                  'dtype': np.complex128},
            random=True,
            eigensolver='dav',
            parallel={'band': 1})
atoms.calc = calc
atoms.get_potential_energy()