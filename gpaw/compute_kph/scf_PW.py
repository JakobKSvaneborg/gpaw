from ase.build import bulk
from gpaw import GPAW,PW, FermiDirac, Davidson
from gpaw.mixer import MixerSum, Mixer
from ase.io import read

atoms = read('cds_scf.in')

mixer = Mixer(beta=0.5, nmaxold=5)

calc = GPAW(mode=PW(800),  h = 0.154, # 0.16 gac 27,27,27, 0.15 gav 30,30,30
            kpts=(8, 8, 8),
            xc='PBE',
            occupations=FermiDirac(0.026),
            nbands = 600,
            eigensolver=Davidson(3),
            mixer=MixerSum(0.02, 5, 100),
            symmetry={'point_group': False, 'time_reversal': False},
            convergence={'energy': 1e-6, 'density': 1e-6, 'bands': 130, 'eigenstates': 4e-07},
            txt='scf_PW.txt')
atoms.calc = calc
atoms.get_potential_energy()

calc.write("scf_PW.gpw", 'all')



