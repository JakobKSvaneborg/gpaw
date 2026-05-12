# creates: al-free-electron.png
# simple free electron calculator
from ase.build import bulk
from ase.calculators.test import FreeElectrons

a = bulk('Al')
kpts = {'path': 'GXWLGK', 'npoints': 100}

# Simple FreeElectron model calculator
a.calc = FreeElectrons(nvalence=3,
                       kpts=kpts)
a.get_potential_energy()
bs = a.calc.band_structure()
bs.plot(emax=10, filename='al-free-electron.png')

from gpaw import GPAW, PW
# calc the self-consistent electron density
a.calc = GPAW(kpts=(3, 3, 3), mode=PW(200), txt='Al.txt')
a.get_potential_energy()
# band-structure calculation for a fixed density
calc = a.calc.fixed_density(
    kpts=kpts,
    symmetry='off',
    nbands=-10,
    convergence={'bands': -5})

bs = calc.band_structure()
bs.plot(emax=10, filename='al-dft.png')
