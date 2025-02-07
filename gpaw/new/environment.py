from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.poisson import PoissonSolver as make_poisson_solver


class Environment:
    def create_poisson_solver(self, grid, *, xp, **kwargs) -> PoissonSolver:
        solver = make_poisson_solver(**kwargs, xp=xp)
        solver.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(solver)

    def update1(self, nt_r):
        pass

    def update2(self, nt_r, vHt_r, vt_sr):
        return 0.0
