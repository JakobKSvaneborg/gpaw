from gpaw import GPAW, PW
from ase.build import bulk
from ase.calculators.dftd3 import DFTD3
from ase.filters import StrainFilter
from ase.optimize.bfgs import BFGS

# This script will optimize lattice constant of metallic lithium
for xc in ['LDA', 'PBE', 'DFTD3']:
    Li_metal = bulk('Li', crystalstructure='bcc', a=3.3)

    if xc == 'DFTD3':
        dft = GPAW(mode=PW(500),
                   kpts=(8, 8, 8),
                   nbands=-10,
                   txt=f'Li-metal-{xc}.log',
                   xc='PBE')
        calc = DFTD3(dft=dft, xc='PBE')
    else:
        calc = GPAW(mode=PW(500),
                    kpts=(8, 8, 8),
                    nbands=-10,
                    txt=f'Li-metal-{xc}.log',
                    xc=xc)

    Li_metal.calc = calc

    sf = StrainFilter(Li_metal, mask=[1, 1, 1, 0, 0, 0])
    opt = BFGS(sf, trajectory=f'Li-metal-{xc}.traj')
    opt.run(fmax=0.01)
