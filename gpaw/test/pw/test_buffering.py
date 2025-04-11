from gpaw import GPAW
from ase.build import molecule
import numpy as np

atoms = molecule('C60')
atoms.center(vacuum=2)
calc = GPAW(mode={'name': 'pw',
                  'ecut': 600,
                  'dtype': np.complex128},
            random=True,
            eigensolver='dav',
            convergence={'maximum iterations': 3},
            parallel={'band': 2})
atoms.calc = calc
atoms.get_potential_energy()