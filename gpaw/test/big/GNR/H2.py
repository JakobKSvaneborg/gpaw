from ase.build import molecule
from ase.optimize import QuasiNewton

from gpaw import GPAW

H2 = molecule('H2', cell=(10, 10, 10))
H2.center()
calc = GPAW(mode='fd')
H2.calc = calc
dyn = QuasiNewton(H2, trajectory='H2.traj')
dyn.run(fmax=0.05)
