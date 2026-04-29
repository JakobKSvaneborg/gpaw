# web-page: si-soc-bs.png
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac
import numpy as np

# Non-collinear ground state calculation:
si = bulk('Si', 'diamond', 5.43)
si.calc = GPAW(mode=PW(400),
               xc='LDA',
               magmoms=np.zeros((2, 3)),
               soc=True,
               kpts=(8, 8, 8),
               symmetry='off',
               occupations=FermiDirac(0.01))
si.get_potential_energy()

bp = si.cell.bandpath('LGX', npoints=100)
bp.plot()

# Restart from ground state and fix density:
calc2 = si.calc.fixed_density(
    nbands=16,
    basis='dzp',
    symmetry='off',
    kpts=bp,
    convergence={'bands': 8})

bs = calc2.band_structure()
bs = bs.subtract_reference()

# Zoom in on VBM:
bs.plot(filename='si-soc-bs.png', show=True, emin=-1.0, emax=0.5)
