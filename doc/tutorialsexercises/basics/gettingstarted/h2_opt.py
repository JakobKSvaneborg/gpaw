# creates: h2.gpaw.traj
from ase import Atoms
from gpaw.calculator import GPAW
from ase.optimize import QuasiNewton

d = 1.0
a = 6.0

atoms = Atoms('H2',
              positions=[(0, 0, 0),
                         (0, 0, d)],
              cell=(a, a, a))
calc = GPAW(mode='pw', txt='h2.txt')

atoms.calc = calc

opt = QuasiNewton(atoms, trajectory='h2.gpaw.traj')

opt.run(fmax=0.05)
