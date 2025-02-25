import ase.io as io
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from gpaw import GPAW
import json

infile = 'unrelaxed.json'
outfile = 'relaxed.json'
param_file = 'params.json'
fmax = 0.01
d3 = False
fixcell = True

if fixcell:
    # take fast optimization with LCAO
    param_key = 'fast_forces'
else:
    # calculate accurate stresses with PW
    param_key = 'accurate_stresses'

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

if fixcell:
    # only optimize positions of the atoms
    opt = BFGS(atoms, logfile=logfile, trajectory=trajectory)
    # --- literalinclude full-opt-start ---
else:
    # --- literalinclude full-opt-else ---
    # setup full relaxation
    # set unit cell filter
    ucf = UnitCellFilter(atoms)
    opt = BFGS(ucf, logfile=logfile, trajectory=trajectory)
    # --- literalinclude full-opt-end ---
# run the optimization until forces are smaller than fmax
opt.run(fmax=fmax)

# write out relaxed structur
io.write(outfile, atoms)
