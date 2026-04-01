from ase.build import bulk
from gpaw.elph import Supercell

atoms = bulk('Si', 'diamond', a=5.431)

sc = Supercell(atoms, supercell_name="supercell", supercell=(3, 3, 3))
calcdict = {'kpts': (4, 4, 4), 'basis': 'dzp'}
sc.calculate_supercell_matrix(calcdict, fd_name='elph')
