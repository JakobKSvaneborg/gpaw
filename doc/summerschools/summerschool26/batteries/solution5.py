from gpaw import GPAW, PW
from ase import Atoms
from ase.optimize.bfgs import BFGS
import numpy as np
from ase.filters import StrainFilter
from gpaw.new.extensions import D3

ccdist = 1.39
a = ccdist * np.sqrt(3)
c = 3.7
for xc in ['LDA', 'PBE', 'DFTD3']:
    calcname = f'Li-C6-{xc}'
    Li_gra = Atoms(
        'C6Li',
        positions=[(0, 0, 0),
                   (0, ccdist, 0),
                   (a, 0, 0),
                   (-a, 0, 0),
                   (-a / 2, -ccdist / 2, 0),
                   (a / 2, -ccdist / 2, 0),
                   (0, -ccdist, c / 2)],
        cell=([1.5 * a, -1.5 * ccdist, 0],
              [1.5 * a, 1.5 * ccdist, 0],
              [0, 0, c]),
        pbc=(1, 1, 1))

    params = dict(
        mode=PW(500),
        kpts=(5, 5, 6),
        txt=calcname + '.log')
    if xc == 'DFTD3':
        dft = GPAW(xc='PBE', extensions=[D3()], **params)
    else:
        calc = GPAW(xc=xc, **params)

    Li_gra.calc = calc  # Connect system and calculator

    sf = StrainFilter(Li_gra, mask=[1, 1, 1, 0, 0, 0])
    opt = BFGS(sf, trajectory=calcname + '.traj')
    opt.run(fmax=0.01)
