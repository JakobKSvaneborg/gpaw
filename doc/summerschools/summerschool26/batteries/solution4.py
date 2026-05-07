import numpy as np
from ase import Atom
from ase.filters import StrainFilter
from ase.io import read
from ase.lattice.hexagonal import Graphene
from ase.optimize.bfgs import BFGS
from gpaw import GPAW, PW
from gpaw.new.extensions import D3

energies = {}
for xc in ['LDA', 'PBE', 'DFTD3']:
    ccdist = 1.40
    layerdist = 3.7

    a = ccdist * np.sqrt(3)
    c = layerdist

    calcname = f'Li-C8-{xc}'
    # We will require a larger cell, to accomodate the Li
    Li_gra = Graphene('C', size=(2, 2, 1), latticeconstant={'a': a, 'c': c})
    Li_gra.append(Atom('Li', (a / 2, ccdist / 2, layerdist / 2)))

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

    e_Li_gra = Li_gra.get_potential_energy()
    energies[xc] = e_Li_gra

for xc, e_Li_gra in energies.items():
    metal = read(f'Li-metal-{xc}.traj')
    e_Li = metal.get_potential_energy() / len(metal)
    gra = read(f'graphite-{xc}.traj')
    e_C8 = 8 * gra.get_potential_energy() / len(gra)
    intercalation_energy = e_Li_gra - (e_Li + e_C8)
    print(f'Intercalation energy: {intercalation_energy:.2f} eV ({xc})')
    # ref = {'LDA': -0.35}[xc]
    # assert abs(intercalation_energy - ref) < 0.01, ref
