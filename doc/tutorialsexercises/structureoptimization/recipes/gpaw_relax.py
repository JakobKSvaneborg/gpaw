import numpy as np
import argparse
import ase.io as io
from ase.optimize import BFGS, QuasiNewton
from gpaw import GPAW
import json

def fixcell_relax(atoms,
                  param_file='params.json',
                  param_key='fast_forces', 
                  fmax=0.01,
                  d3=False):

    with open(param_file, 'r') as f:
        params = json.load(f)

    calculator_params = params[param_key]["calculator"]
    # remove 'name' from calculator params
    calculatorname = calculator_params.pop('name')

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

    return atoms

def full_relax(atoms,
               param_file='params.json',
               param_key='fast_forces', 
               fmax=0.01,
               d3=False):

    from ase.filters import UnitCellFilter

    with open(param_file, 'r') as f:
        params = json.load(f)

    calculator_params = params[param_key]["calculator"]
    # remove 'name' from calculator params
    calculatorname = calculator_params.pop('name')

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

    # set unit cell filter
    ucf = UnitCellFilter(atoms) 
    opt = BFGS(ucf, logfile=logfile, trajectory=trajectory)

    # run the optimization until forces are smaller than fmax
    opt.run(fmax=fmax)

    return atoms

def main():
    parser = argparse.ArgumentParser(description='Relax given structure')
    parser.add_argument('-infile', help='infile', type=str, default='unrelaxed.json')
    parser.add_argument('-key', help='param key', type=str, default='fast_forces')
    parser.add_argument('-outfile', help='outfile', type=str, default='relaxed.json')
    parser.add_argument('-fixcell', help='fixcell relaxation', default=False)
    args = parser.parse_args()

    atoms = io.read(args.infile)

    if args.fixcell:
        atoms = fixcell_relax(atoms, param_key=args.key) 
    else:
        atoms = full_relax(atoms, param_key=args.key) 
        
    io.write(args.outfile, atoms)

if __name__ == "__main__":
    main()
