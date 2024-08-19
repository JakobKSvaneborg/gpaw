from gpaw import GPAW


def optimize_cell(atoms, calculator):
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS

    atoms.calc = GPAW(**calculator)
    opt = BFGS(FrechetCellFilter(atoms), trajectory='opt.traj',
               logfile='opt.log')
    opt.run(fmax=0.01)
    return atoms

# end-optimize-cell-snippet

def groundstate(atoms, calculator):
    from pathlib import Path
    atoms.calc = GPAW(**calculator)
    atoms.get_potential_energy()
    path = Path('groundstate.gpw')
    atoms.calc.write(path)
    return path

# --- literalinclude-divider-2 ---

def bandpath(atoms):
    return atoms.cell.bandpath(npoints=100)


def bandstructure(gpw, bandpath):
    gscalc = GPAW(gpw)
    atoms = gscalc.get_atoms()
    calc = gscalc.fixed_density(
        kpts=bandpath, symmetry='off', txt='bs.txt')
    bs = calc.band_structure()
    return bs
