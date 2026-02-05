from ase.build import bulk
from gpaw import GPAW,PW, FermiDirac
from gpaw.mixer import Mixer
from ase.io import read

atoms = read('cds_scf.in')


mixer = Mixer(beta=0.5, nmaxold=5)


calc = GPAW(mode='lcao', basis='dzp', h = 0.122, # 0.13 virkede før jeg indsatte mixer og hævede konvergenskriterier.
            kpts=(8, 8, 8),
            xc='PBE',
            occupations=FermiDirac(0.026),
            mixer=mixer,
            nbands = 34, # was not here before
            symmetry={'point_group': False, 'time_reversal': False},
            convergence={'energy': 1e-6, 'density': 1e-6, 'bands': 100, 'eigenstates': 4e-08}, # energi was 1e-5 before.
            txt='scf.txt')
atoms.calc = calc
atoms.get_potential_energy()

calc.write("scf.gpw", 'all')

