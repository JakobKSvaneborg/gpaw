from ase.build import molecule

from gpaw import GPAW
from gpaw.odd.sic import SIC

n = molecule('N')
n.center(vacuum=3)
calc = GPAW(mode='fd',
            xc=SIC(nspins=2),
            h=0.2,
            convergence=dict(eigenvalues=1e-5, density=1e-3),
            txt='N.txt')
n.calc = calc
e1 = n.get_potential_energy()

n2 = molecule('N2')
n2.center(vacuum=3)
calc = GPAW(mode='fd',
            xc=SIC(nspins=1),
            h=0.2,
            convergence=dict(eigenvalues=1e-5, density=1e-3),
            txt='N2.txt')
n2.calc = calc
e2 = n2.get_potential_energy()

assert abs(e2 - 2 * e1 - -4.5) < 0.1
