from gpaw.new.environment import Environment
from gpaw.solvation.poisson import WeightedFDPoissonSolver
from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper


class Solvation(Environment):
    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions=None,
                 setups, grid, fracpos_ac, log):
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions or []
        finegd = grid._gd
        self.grid = grid
        self.cavity.set_grid_descriptor(finegd)
        self.dielectric.set_grid_descriptor(finegd)
        for ia in self.interactions:
            ia.set_grid_descriptor(finegd)
        self.cavity.allocate()
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        from ase import Atoms
        atoms = Atoms([setup.symbol for setup in setups],
                      scaled_positions=fracpos_ac,
                      cell=grid.cell,
                      pbc=grid.pbc)
        self.cavity.update_atoms(atoms, log)
        for ia in self.interactions:
            ia.update_atoms(atoms)

    def create_poisson_solver(self, grid, *, xp, **kwargs) -> PoissonSolver:
        psolver = WeightedFDPoissonSolver()
        psolver.set_dielectric(self.dielectric)
        psolver.set_grid_descriptor(self.grid._gd)
        return PoissonSolverWrapper(psolver)
