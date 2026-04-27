# web-page: ACF.dat
import subprocess
from gpaw.utilities.bader import read_bader_charges
subprocess.call('bader -p all_atom -p atom_index density.cube'.split())
charges = read_bader_charges()
assert abs(sum(charges) - 10) < 0.0001
assert abs(charges[1] - 0.425) < 0.005
assert abs(charges[2] - 0.425) < 0.005
