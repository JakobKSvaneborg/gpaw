import numpy as np
from ase.filters import StrainFilter
from ase.lattice.hexagonal import Graphite
from ase.optimize.bfgs import BFGS
from gpaw import GPAW, PW
from gpaw.new.extensions import D3

ccdist = 1.41
layerdist = 3.21
a = ccdist * np.sqrt(3)
c = 2 * layerdist
gra = Graphite('C', latticeconstant={'a': a, 'c': c})

for xc in ['LDA', 'PBE', 'DFTD3']:
    params = dict(
        mode=PW(500),
        kpts=(10, 10, 6),
        txt=f'graphite-{xc}.txt')
    if xc == 'DFTD3':
        calc = GPAW(xc='PBE', extensions=[D3()], **params)
    else:
        calc = GPAW(xc=xc, **params)

    gra.calc = calc  # Connect system and calculator

    sf = StrainFilter(gra, mask=[1, 1, 1, 0, 0, 0])
    opt = BFGS(sf)
    opt.run(fmax=0.01)
