from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw import PW

a = 3.567
atoms = bulk('C', 'diamond', a=a)

calc = GPAW(mode='lcao',
            basis='dzp',
            kpts={'size': (8, 8, 8), 'gamma': True},
            xc='LDA',
            occupations=FermiDirac(0.001),
            txt='C_lcao_groundstate.txt')

atoms.calc = calc
atoms.get_potential_energy()
calc.write('C_lcao_groundstate.gpw', mode='all')
