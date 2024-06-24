from ase import optimize
from ase.build import molecule
from ase.vibrations.infrared import Infrared
from gpaw import GPAW, FermiDirac
from gpaw.utilities.adjust_cell import adjust_cell

h = 0.20

atoms = molecule('H2O')
adjust_cell(atoms, 4, h=h)

# relax the molecule
calc = GPAW(mode='fd', xc='PBE', h=h, occupations=FermiDirac(width=0.1))
atoms.calc = calc

dyn = optimize.FIRE(atoms)
dyn.run(fmax=0.01)
atoms.write('relaxed.traj')

# finite displacement for vibrations
atoms.calc = calc.new(symmetry={'point_group': False})
ir = Infrared(atoms, name='ir')
ir.run()
