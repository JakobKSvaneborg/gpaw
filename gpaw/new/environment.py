from __future__ import annotations
import numpy as np
from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.poisson import PoissonSolver as make_poisson_solver
from gpaw.core import UGArray, UGDesc, PWArray
from ase.units import Ha
from gpaw.new.ibzwfs import IBZWaveFunctions


class Environment:
    def __init__(self, natoms: int):
        self.natoms = natoms
        self.charge = 0.0

    def create_poisson_solver(self, grid, *, xp, **kwargs) -> PoissonSolver:
        solver = make_poisson_solver(**kwargs, xp=xp)
        solver.set_grid_descriptor(grid._gd)
        return PoissonSolverWrapper(solver)

    def check_convergence(self,
                          ibzwfs: IBZWaveFunctions,
                          log) -> bool:
        return True

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             occ_calc,
                             mixer) -> None:
        pass

    def update1(self, nt_r):
        pass

    def update1pw(self, nt_g):
        pass

    def update2(self, nt_r, vHt_r, vt_sr):
        return 0.0

    def forces(self, nt_r, vHt_r):
        return np.zeros((self.natoms, 3))


class Jellium(Environment):
    def __init__(self,
                 jellium,
                 natoms: int,
                 grid: UGDesc):
        super().__init__(natoms)
        self.grid = grid
        self.charge = jellium.charge
        self.charge_x: UGArray | PWArray | None = grid.zeros()
        jellium.add_charge_to(self.charge_x.data)

    def update1(self, nt_r: UGArray) -> None:
        assert self.charge_x is not None
        nt_r.data += self.charge_x.data

    def update1pw(self, nt_g: PWArray | None) -> None:
        if isinstance(self.charge_x, UGArray):
            charge_r = self.charge_x.gather()
            if nt_g is not None:
                self.charge_x = charge_r.fft(pw=nt_g.desc)
            else:
                self.charge_x = None
        if nt_g is None:
            return
        assert self.charge_x is not None
        nt_g.data += self.charge_x.data


class FixedPotentialJellium(Jellium):
    def __init__(self,
                 jellium,
                 natoms: int,
                 grid: UGDesc,
                 fermi_level: float):
        super().__init__(jellium, natoms, grid)
        self.fixed_fermi_level = fermi_level / Ha
        self.history: list[tuple[float, float]] = []

    def check_convergence(self,
                          ibzwfs: IBZWaveFunctions,
                          log) -> bool:
        fl = ibzwfs.fermi_level
        log(f'charge: {self.charge:.6f} |e|, Fermi-level: {fl * Ha:.3f} eV')
        tol = 0.001 / Ha
        return abs(fl - self.fixed_fermi_level) <= tol

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             occ_calc,
                             mixer) -> None:
        fl = ibzwfs.fermi_level
        self.history.append((self.charge, fl))
        if len(self.history) == 1:
            area = abs(np.linalg.det(self.grid.cell_cv[:2, :2]))
            dc = -(fl - self.fixed_fermi_level) * area * 0.02
        else:
            (c0, fl0), (c1, fl1) = self.history[-2:]
            c = c0 + (self.fixed_fermi_level - fl0) / (fl1 - fl0) * (c1 - c0)
            dc = c - c1
            if abs(dc) > abs(c0 - c1):
                dc *= abs((c0 - c1) / dc)
        new_charge = self.charge + dc
        if self.charge_x is not None:
            self.charge_x.data *= new_charge / self.charge
        self.charge = new_charge
        ibzwfs.nelectrons += dc
        ibzwfs.calculate_occs(occ_calc)
        mixer.reset()
