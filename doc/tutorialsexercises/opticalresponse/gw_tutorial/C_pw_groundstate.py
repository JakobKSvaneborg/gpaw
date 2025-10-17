from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw import PW

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode=PW(ecut=500),
            kpts={'size': (8, 8, 8), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.0),
            txt='C_pw_groundstate.txt')

atoms.calc = calc
atoms.get_potential_energy()
calc.diagonalize_full_hamiltonian()
calc.write('C_pw_groundstate.gpw', mode='all')
