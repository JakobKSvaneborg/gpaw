from ase.build import molecule
from ase.io import write
from ase.units import Bohr
from gpaw import GPAW
from gpaw.analyse.hirshfeld import HirshfeldPartitioning

atoms = molecule('H2O')
atoms.center(vacuum=3.5)
atoms.calc = GPAW(mode='fd', h=0.17, txt='h2o.txt', legacy_gpaw=True)
atoms.get_potential_energy()

# write Hirshfeld charges out
hf = HirshfeldPartitioning(atoms.calc)
for atom, charge in zip(atoms, hf.get_charges()):
    atom.charge = charge
# atoms.write('Hirshfeld.traj') # XXX Trajectory writer needs a fix
atoms.copy().write('Hirshfeld.traj')

# create electron density cube file ready for bader
rho = atoms.calc.get_all_electron_density(gridrefinement=4)
write('density.cube', atoms, data=rho * Bohr**3)
