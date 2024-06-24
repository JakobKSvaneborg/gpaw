import numpy as np
from ase.units import Bohr, Ha
from ase.build import molecule
from gpaw.utilities.adjust_cell import adjust_cell
from gpaw import GPAW
from gpaw.external import static_polarizability


atoms = molecule('H2O')
adjust_cell(atoms, border=3)
atoms.calc = GPAW(mode='fd', txt=None)

alpha_cc = static_polarizability(atoms)
print('Polarizability tensor (units Angstrom^3):')
print(alpha_cc * Bohr * Ha)

w, v = np.linalg.eig(alpha_cc)
print('Eigenvalues', w * Bohr * Ha)
print('Eigenvectors', v)
print('average polarizablity', w.sum() * Bohr * Ha / 3, 'Angstrom^3')
