# creates: h2o.gpaw.traj
from ase import Atoms
from gpaw.calculator import GPAW
from ase.optimize import QuasiNewton

a = 6.0

atoms = Atoms('OH2',
              positions=[(0, 0, 0),
                         (0, 0.6, -0.75),
                         (0, 0.6, +0.75)],
              cell=(a, a, a))
atoms.center()

calc = GPAW(mode='pw', txt='h2o.txt')
atoms.calc = calc

opt = QuasiNewton(atoms, trajectory='h2o.gpaw.traj')
opt.run(fmax=0.05)
