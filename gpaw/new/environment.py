import numpy as np
from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.poisson import PoissonSolver as make_poisson_solver
from gpaw.core import UGArray
from ase.units import Ha


class Environment:
    def __init__(self, natoms: int):
        self.natoms = natoms
        self.charge = 0.0
        self.fixed_fermi_level = None

    def create_poisson_solver(self, grid, *, xp, **kwargs) -> PoissonSolver:
        solver = make_poisson_solver(**kwargs, xp=xp)
        solver.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(solver)

    def update1(self, nt_r, c):
        pass

    def update2(self, nt_r, vHt_r, vt_sr):
        return 0.0

    def forces(self, nt_r, vHt_r):
        return np.zeros((self.natoms, 3))


class Jellium(Environment):
    def __init__(self, jellium, natoms, grid, fermi_level):
        super().__init__(natoms)
        if fermi_level is not None:
            self.fixed_fermi_level = fermi_level / Ha
        self.jellium = jellium
        self.grid = grid
        self.charge = jellium.charge
        self.charge_g = None

    def update1(self, nt_x) -> None:
        if isinstance(nt_x, UGArray):
            self.jellium.add_charge_to(nt_x.data)
            return
        nt_g = nt_x
        if self.charge_g is None:
            charge_r = self.grid.zeros()
            self.jellium.add_charge_to(charge_r.data)
            self.charge_g = charge_r.fft(pw=nt_g.desc)
            self.charge_g.data *= 1.0 / self.charge_g.integrate()
        nt_g.data -= self.charge_g.data * self.charge
