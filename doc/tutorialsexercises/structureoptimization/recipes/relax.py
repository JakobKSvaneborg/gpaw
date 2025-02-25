from ase.build import bulk
from ase.optimize import BFGS
from gpaw import GPAW
# --- literalinclude import-end ---
from ase.filters import FrechetCellFilter
# --- literalinclude import-filter ---
from ase import io
import json


def main():
    # a0 = 5.421
    a = 5.6
    param_file = 'params.json'
    outfile = 'relaxed.json'

    atoms = bulk('Si', 'fcc', a=a)
    atoms.rattle(0.05)

    with open(param_file, 'r') as f:
        params = json.load(f)

    atoms = relax(atoms, params, fixcell=True)
    atoms = relax(atoms, params, fixcell=False)

    # write out relaxed structure
    io.write(outfile, atoms)


# --- literalinclude relax-start ---
def relax(atoms, params, fmax=0.01,
          d3=False, fixcell=True,
          logname='opt.log',
          trajname='opt.traj'):

    if fixcell:
        # take fast optimization with LCAO
        param_key = 'fast_forces'
    else:
        # calculate accurate stresses with PW
        param_key = 'accurate_stresses'

    calculator_params = params[param_key]["calculator"]
    # remove 'name' from calculator params
    assert 'gpaw' == calculator_params.pop('name')

    # set DFT calculator
    calc_dft = GPAW(**calculator_params)

    # magnetize atoms
    atoms.set_initial_magnetic_moments(len(atoms) * [1])
    # non-magnetic calculation:
    # atoms.set_initial_magnetic_moments(len(atoms) * [0])

    # optionally include van der Waals DFT-D3
    if d3:
        from ase.calculators.dftd3 import DFTD3
        calc = DFTD3(dft=calc_dft)
    else:
        calc = calc_dft

    # set calculator
    atoms.calc = calc

    # set configuration to be optimized
    if fixcell:
        # only optimize positions of the atoms
        opt_conf = atoms
    else:
        # setup full relaxation
        # set unit cell filter
        opt_conf = FrechetCellFilter(atoms)

    # setup optimizer
    # specify logfile and trajectory file names
    opt = BFGS(opt_conf, logfile=logname, trajectory=trajname)
    # run the optimization until forces are smaller than fmax
    opt.run(fmax=fmax)

    return atoms
# --- literalinclude relax-end ---


if __name__ == "__main__":
    main()
