import numpy as np
from ase.calculators.dftd3 import DFTD3
from ase.filters import StrainFilter
from ase.lattice.hexagonal import Graphite
from ase.optimize.bfgs import BFGS
from gpaw import GPAW, PW

ccdist = 1.41
layerdist = 3.21
a = ccdist * np.sqrt(3)
c = 2 * layerdist
gra = Graphite('C', latticeconstant={'a': a, 'c': c})


for xc in ['LDA', 'PBE', 'DFTD3']:
    if xc == 'DFTD3':
        dft = GPAW(mode=PW(500),
                   kpts=(10, 10, 6),
                   xc='PBE',
                   txt=f'graphite-{xc}.txt')
        calc = DFTD3(dft=dft, xc='PBE')
    else:
        calc = GPAW(mode=PW(500),
                    kpts=(10, 10, 6),
                    xc=xc,
                    txt=f'graphite-{xc}.txt')

    gra.calc = calc  # Connect system and calculator

    sf = StrainFilter(gra, mask=[1, 1, 1, 0, 0, 0])
    opt = BFGS(sf)
    opt.run(fmax=0.01)
