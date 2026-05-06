from gpaw import GPAW, PW
from ase import Atom
from ase.optimize.bfgs import BFGS
import numpy as np
from ase.lattice.hexagonal import Graphene
from ase.filters import StrainFilter

for xc in ['LDA', 'PBE', 'DFTD3']:
    ccdist = 1.40
    layerdist = 3.7

    a = ccdist * np.sqrt(3)
    c = layerdist

    calcname = f'Li-C8-{xc}'
    # We will require a larger cell, to accomodate the Li
    Li_gra = Graphene('C', size=(2, 2, 1), latticeconstant={'a': a, 'c': c})
    Li_gra.append(Atom('Li', (a / 2, ccdist / 2, layerdist / 2)))

    if xc == 'DFTD3':
        dft = GPAW(mode=PW(500),
                   kpts=(5, 5, 6),
                   xc='PBE',
                   txt=calcname + '.log')
        calc = DFTD3(dft=dft, xc='PBE')
    else:
        calc = GPAW(mode=PW(500), kpts=(5, 5, 6), xc=xc, txt=calcname + '.log')

    Li_gra.calc = calc  # Connect system and calculator

    sf = StrainFilter(Li_gra, mask=[1, 1, 1, 0, 0, 0])
    opt = BFGS(sf, trajectory=calcname + '.traj')
    opt.run(fmax=0.01)

    e_Li_gra = Li_gra.get_potential_energy()
    e_Li = Li_metal.get_potential_energy() / len(Li_metal)
    e_C8 = 8 * gra.get_potential_energy() / len(gra)
    intercalation_energy = e_Li_gra - (e_Li + e_C8)
    print(f'Intercalation energy: {intercalation_energy:.2f} eV')
