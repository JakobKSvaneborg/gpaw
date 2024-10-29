from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.poisson import PoissonSolver as make_poisson_solver


class Environment:
    def create_poisson_solver(self, grid, *, xp, **kwargs) -> PoissonSolver:
        solver = make_poisson_solver(**kwargs, xp=xp)
        solver.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(solver)
