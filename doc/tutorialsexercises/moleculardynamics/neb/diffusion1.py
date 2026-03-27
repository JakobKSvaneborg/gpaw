from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from gpaw import GPAW, PW

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)

# Make sure the structure is correct:
# view(slab)

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
# print(mask)
slab.set_constraint(FixAtoms(mask=mask))

slab.calc = GPAW(mode=PW(ecut=500),
                 kpts=(4, 4, 1),
                 txt='initial.txt')

# Initial state:
qn = QuasiNewton(slab, trajectory='initial.traj')
qn.run(fmax=0.05)

# Final state:
slab[-1].x += slab.get_cell()[0, 0] / 2
slab.calc = slab.calc.new(txt='final.txt')
qn = QuasiNewton(slab)
qn.run(fmax=0.05)

qn = QuasiNewton(slab, trajectory='final.traj')
qn.run(fmax=0.05)
