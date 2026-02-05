from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw.elph import DisplacementRunner
from ase.io import read

atoms = read('cds_scf.in')

calc = GPAW(mode='lcao', h=0.18, basis='dzp',
            kpts=(4, 4, 4),
            xc='PBE',
            occupations=FermiDirac(0.026),
            symmetry={'point_group': False},
            convergence={'energy': 2e-5, 'density': 1e-5},
            txt='displacement.txt')

elph = DisplacementRunner(atoms=atoms, calc=calc,
                          supercell=(4, 4, 4), name='elph',
                          calculate_forces=True)  # 5,5,5 was doable
elph.run()
