import numpy as np
from ase.units import Bohr, Ha
from ase.build import molecule
from gpaw import GPAW
from gpaw.external import static_polarizability


atoms = molecule('H2O')
atoms.center(vacuum=3.0)
atoms.calc = GPAW(mode='fd', legacy_gpaw=True)

alpha_cc = static_polarizability(atoms)
print('Polarizability tensor (units Angstrom^3):')
print(alpha_cc * Bohr * Ha)

w, v = np.linalg.eig(alpha_cc)
print('Eigenvalues', w * Bohr * Ha)
print('Eigenvectors', v)
ave = w.sum() * Bohr * Ha / 3
print('average polarizablity', ave, 'Angstrom^3')
assert abs(ave - 1.54) < 0.01
