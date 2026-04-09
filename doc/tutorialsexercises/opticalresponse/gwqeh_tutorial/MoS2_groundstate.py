from ase.build import mx2
from gpaw import GPAW, PW, FermiDirac

# Create MoS2 monolayer structure
structure = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                size=(1, 1, 1), vacuum=7.5)

# Set up GPAW calculator
calc = GPAW(mode=PW(400),
            parallel={'domain': 1},
            xc='LDA',
            kpts={'size': (9, 9, 1), 'gamma': True},
            occupations=FermiDirac(0.01),
            txt='MoS2_groundstate.txt')

structure.calc = calc
structure.get_potential_energy()

# Diagonalize to get all bands
calc.diagonalize_full_hamiltonian()
calc.write('MoS2_fulldiag.gpw', 'all')
