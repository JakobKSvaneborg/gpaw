import numpy as np
from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.poisson import PoissonSolver as make_poisson_solver


class Environment:
    def __init__(self, natoms: int):
        self.natoms = natoms

    def create_poisson_solver(self, grid, *, xp, **kwargs) -> PoissonSolver:
        solver = make_poisson_solver(**kwargs, xp=xp)
        solver.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(solver)

    def update1(self, nt_r):
        pass

    def update2(self, nt_r, vHt_r, vt_sr):
        return 0.0

    def forces(self, nt_r, vHt_r):
        return np.zeros((self.natoms, 3))


class Jellium(Environment):
    def __init__(self, jellium, natoms):
        super().__init__(natoms)
        self.jellium = jellium

    def update1(self, nt_r):
        self.jellium.add_charge_to(nt_r.data)
