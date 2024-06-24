from gpaw.utilities.adjust_cell import adjust_cell
from ase import Atoms

d = 0.74
a = 6.0
atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, d)])
# set the amount of vacuum at least to 4 Å
# and ensure a grid spacing of h=0.2
adjust_cell(atoms, 4.0, h=0.2)
