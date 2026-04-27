from pathlib import Path
from gpaw import GPAW
import taskblaster as tb


def optimize_cell(atoms, calculator):
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGS
    from gpaw import GPAW

    atoms.calc = GPAW(**calculator)
    opt = BFGS(FrechetCellFilter(atoms), trajectory='opt.traj',
               logfile='opt.log')
    opt.run(fmax=0.01)
    return atoms

def groundstate(atoms, calculator):
    atoms.calc = GPAW(**calculator)
    atoms.get_potential_energy()
    path = Path('groundstate.gpw')
    atoms.calc.write(path)
    return path

def bandpath(atoms):
    return atoms.cell.bandpath(npoints=100)


def bandstructure(gpw, bandpath):
    gscalc = GPAW(gpw)
    atoms = gscalc.get_atoms()
    calc = gscalc.fixed_density(
        kpts=bandpath, symmetry='off', txt='bs.txt')
    bs = calc.band_structure()
    return bs


@tb.workflow
class MaterialsWorkflow:
    atoms = tb.var()
    calculator = tb.var()

    @tb.task
    def relax(self):
        return tb.node(
            'optimize_cell',
            atoms=self.atoms,
            calculator=self.calculator)

    @tb.task
    def groundstate(self):
        return tb.node(
            'groundstate',
            atoms=self.relax,
            calculator=self.calculator)

    @tb.task
    def bandpath(self):
        return tb.node('bandpath', atoms=self.relax)

    @tb.task
    def bandstructure(self):
        return tb.node(
            'bandstructure',
            gpw=self.groundstate,
            bandpath=self.bandpath)


def asebulk(symbol):
    from ase.build import bulk
    return bulk(symbol)


@tb.workflow
class ParametrizableMaterialsWorkflow:
    symbol = tb.var()
    calculator = tb.var()

    @tb.task
    def atoms(self):
        return tb.node('asebulk', symbol=self.symbol)

    @tb.subworkflow
    def compute(self):
        return MaterialsWorkflow(atoms=self.atoms, calculator=self.calculator)


@tb.dynamical_workflow_generator_task
def parametrize_materials_workflow(calculator):

    material_symbols = [
        'Al', 'Si', 'Ti', 'Cu', 'Ag', 'Au', 'Pd', 'Pt',
    ]

    for symbol in material_symbols:
        yield f'mat-{symbol}', ParametrizableMaterialsWorkflow(
            symbol=symbol, calculator=calculator)
