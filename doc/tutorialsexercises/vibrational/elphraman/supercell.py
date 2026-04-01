from ase.io import read
from gpaw import GPAW, FermiDirac
from gpaw.elph import Supercell


def get_lcao_calc(gpt, kpts):
    return GPAW(mode='lcao',
                parallel={'sl_auto': True, 'augment_grids': True,
                          'domain': 1, 'band': 1},
                xc='PBE',
                gpts=gpt,
                kpts=kpts,
                occupations=FermiDirac(0.02),
                convergence={'density': 1e-6},
                symmetry={'point_group': False},
                txt='supercell.txt')


atoms = read("MoS2_2H_relaxed_PBE.json")
atoms_N = atoms * (3, 3, 2)
calc = get_lcao_calc((60, 60, 160), (2, 2, 2))

atoms_N.calc = calc
atoms_N.get_potential_energy()

sc = Supercell(atoms=atoms, supercell=(3, 3, 2))
sc.calculate_supercell_matrix(calc, fd_name='elph')
