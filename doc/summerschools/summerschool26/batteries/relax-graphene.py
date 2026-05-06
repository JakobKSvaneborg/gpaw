import numpy as np
from ase.lattice.hexagonal import Graphite
from gpaw import GPAW, PW
from ase.filters import StrainFilter
from ase.optimize.bfgs import BFGS

ccdist = 1.40
layerdist = 3.33
a = ccdist * np.sqrt(3)
c = 2 * layerdist
gra = Graphite('C', latticeconstant={'a': a, 'c': c})

xc = 'LDA'
calcname = f'graphite-{xc}'
calc = GPAW(mode=PW(500),
            kpts=(10, 10, 6),
            xc=xc,
            txt=calcname + '.log')
gra.calc = calc  # Connect system and calculator
print(gra.get_potential_energy())

sf = StrainFilter(gra, mask=[1, 1, 1, 0, 0, 0])
opt = BFGS(sf, trajectory=calcname + '.traj')
opt.run(fmax=0.01)
