import numpy as np
import ase.io as io
from ase.optimize import BFGS
from gpaw import GPAW
import json

infile = 'unrelaxed.json'
outfile = 'relaxed.json'
param_file = 'params.json'
param_key = 'fast_forces' 
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
atoms.set_initial_magnetic_moments(np.ones(len(atoms), float))

if d3:
    from ase.calculators.dftd3 import DFTD3
    calc = DFTD3(dft=calc_dft)
else:
    calc = calc_dft

atoms.calc = calc

# make logfile and trajectory file objects and attach to optimizer
logfile=open('opt.log','w')
trajectory = io.Trajectory('opt.traj', 'w', atoms)
opt = BFGS(atoms, logfile=logfile, trajectory=trajectory)

# run the optimization until forces are smaller than fmax
opt.run(fmax=fmax)

io.write(outfile, atoms)
