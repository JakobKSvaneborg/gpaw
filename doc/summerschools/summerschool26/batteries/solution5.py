import numpy as np
from ase import Atoms
from ase.filters import StrainFilter
from ase.io import read
from ase.optimize.bfgs import BFGS
from gpaw import GPAW, PW
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
        txt=calcname + '.txt')
    if xc == 'DFTD3':
        calc = GPAW(xc='PBE', extensions=[D3(xc='PBE')], **params)
    else:
        calc = GPAW(xc=xc, **params)

    Li_gra.calc = calc  # Connect system and calculator

    sf = StrainFilter(Li_gra, mask=[1, 1, 1, 0, 0, 0])
    opt = BFGS(sf, trajectory=calcname + '.traj')
    opt.run(fmax=0.01)

for xc in ['LDA', 'PBE', 'DFTD3']:
    metal = read(f'Li-metal-{xc}.traj')
    e_Li = metal.get_potential_energy() / len(metal)
    gra = read(f'graphite-{xc}.traj')
    e_C = gra.get_potential_energy() / len(gra)
    e_Li_gra = read(f'Li-C6-{xc}.traj').get_potential_energy()
    intercalation_energy = e_Li_gra - (e_Li + 6 * e_C)
    print(f'Intercalation energy: {intercalation_energy:.2f} eV ({xc})')
    ref = {'LDA': -0.41,
           'PBE': -0.09,
           'DFTD3': -0.08}[xc]
    assert abs(intercalation_energy - ref) < 0.01, ref
