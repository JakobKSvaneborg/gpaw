from ase.build import bulk
from gpaw import GPAW, FermiDirac
from gpaw import PW

a = 3.567
atoms = bulk('C', 'diamond', a=a)

# Make sure we store all bands
calc = GPAW(mode='lcao',
            basis='dzp',
            kpts={'size': (8, 8, 8), 'gamma': True},
            xc='LDA',
            nbands='nao',
            occupations=FermiDirac(0.0),
            txt='C_lcao_groundstate.txt')

atoms.calc = calc
atoms.get_potential_energy()
calc.write('C_lcao_groundstate.gpw', mode='all')
