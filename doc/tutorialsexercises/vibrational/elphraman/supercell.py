from ase.io import read
from gpaw.elph import Supercell

atoms = read("MoS2_2H_relaxed_PBE.json")
sc = Supercell(atoms, supercell=(3, 3, 2))
calcdict = {'basis': 'dzp', 'kpts': (2, 2, 2)}
sc.calculate_supercell_matrix(calcdict, fd_name='elph')
