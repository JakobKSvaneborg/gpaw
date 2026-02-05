from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.elph import Supercell
from ase.io import read

atoms = read('cds_scf.in')

atoms_N = atoms * (4, 4, 4)

calc = GPAW(mode='lcao', h=0.18, basis='dzp',
            kpts=(4, 4, 4),
            xc='PBE',
            occupations=FermiDirac(0.026),
            symmetry={'point_group': False},
            txt='supercell.txt',
            parallel={'domain': 1, 'band': 1})

atoms_N.calc = calc
atoms_N.get_potential_energy()

sc = Supercell(atoms, supercell=(4, 4, 4))
sc.calculate_supercell_matrix(calc, fd_name='elph')

