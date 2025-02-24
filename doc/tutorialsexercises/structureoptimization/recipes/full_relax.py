import ase.io as io
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from gpaw import GPAW
import json

infile = 'unrelaxed.json'
outfile = 'relaxed.json'
param_file = 'params.json'
param_key = 'accurate_stresses'
fmax = 0.01
d3 = False

atoms = io.read(infile)

with open(param_file, 'r') as f:
    params = json.load(f)

calculator_params = params[param_key]["calculator"]
# remove 'name' from calculator params
assert 'gpaw' == calculator_params.pop('name')

# set DFT calculator
calc_dft = GPAW(**calculator_params)

# magnetize atoms
atoms.set_initial_magnetic_moments(len(atoms) * [1])

# optionally include van der Waals DFT-D3
if d3:
    from ase.calculators.dftd3 import DFTD3
    calc = DFTD3(dft=calc_dft)
else:
    calc = calc_dft

# set calculator
atoms.calc = calc

# make logfile and trajectory file objects and attach to optimizer
logfile = open('opt.log', 'w')
trajectory = io.Trajectory('opt.traj', 'w', atoms)

# --- literalinclude start line ---
# set unit cell filter
ucf = UnitCellFilter(atoms)
opt = BFGS(ucf, logfile=logfile, trajectory=trajectory)
# --- literalinclude end line ---

# run the optimization until forces are smaller than fmax
opt.run(fmax=fmax)

io.write(outfile, atoms)
