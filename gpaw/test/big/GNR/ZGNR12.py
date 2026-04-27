from ase.build import graphene_nanoribbon
from ase.optimize import QuasiNewton

from gpaw import GPAW

GNR = graphene_nanoribbon(12, 1, type='zigzag', vacc=6)
GNR.set_pbc((0, 0, 1))
kpts = (1, 1, 10)
calc = GPAW(mode='fd', kpts=kpts, spinpol=True)
GNR.calc = calc
dyn = QuasiNewton(GNR, trajectory='ZGNR12.traj')
dyn.run(fmax=0.05)
