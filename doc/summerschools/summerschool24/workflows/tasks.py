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

def bandstructure(gpw):
    gscalc = GPAW(gpw)
    atoms = gscalc.get_atoms()
    bandpath = atoms.cell.bandpath(npoints=100)
    bandpath.write('bandpath.json')
    calc = gscalc.fixed_density(
        kpts=bandpath.kpts, symmetry='off', txt='bs.txt')
    bs = calc.band_structure()
    bs.write('bs.json')
    return bs
