from ase.build import bulk
from ase.filters import StrainFilter
from ase.optimize.bfgs import BFGS
from gpaw import GPAW, PW
from gpaw.new.extensions import D3

# This script will optimize lattice constant of metallic lithium
for xc in ['LDA', 'PBE', 'DFTD3']:
    Li_metal = bulk('Li', crystalstructure='bcc', a=3.3)

    params = dict(
        mode=PW(500),
        kpts=(8, 8, 8),
        nbands=-10,
        txt=f'Li-metal-{xc}.txt')

    if xc == 'DFTD3':
        calc = GPAW(xc='PBE', extensions=[D3(xc='PBE')], **params)
    else:
        calc = GPAW(xc=xc, **params)

    Li_metal.calc = calc

    sf = StrainFilter(Li_metal, mask=[1, 1, 1, 0, 0, 0])
    opt = BFGS(sf, trajectory=f'Li-metal-{xc}.traj')
    opt.run(fmax=0.01)
