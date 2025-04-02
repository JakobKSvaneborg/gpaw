from __future__ import annotations

import numpy as np
from ase.units import Bohr
from gpaw.jellium import create_background_charge
from gpaw.new.environment import Environment, FixedPotentialJellium, Jellium
from gpaw.new.poisson import PoissonSolverWrapper
from gpaw.new.solvation import Solvation
from gpaw.core import UGArray


class SJM:
    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions,
                 jelliumregion,
                 target_potential: float | None,  # eV
                 excess_electrons: float = 0.0,
                 tol: float = 0.001):  # eV
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        self.jelliumregion = jelliumregion
        self.target_potential = target_potential
        self.excess_electrons = excess_electrons
        self.tol = tol

    def build(self,
              setups,
              grid,
              relpos_ac,
              log,
              comm,
              nn) -> SJMEnvironment:
        solvation = Solvation(
            cavity=self.cavity,
            dielectric=self.dielectric,
            interactions=self.interactions,
            setups=setups, grid=grid, relpos_ac=relpos_ac,
            log=log, comm=comm, nn=nn)
        h = grid.cell_cv[2, 2] * Bohr
        z1 = relpos_ac[:, 2].max() * h + 3.0
        z2 = self.jelliumregion.get('top', h - 1.0)
        background = create_background_charge(charge=self.excess_electrons,
                                              z1=z1,
                                              z2=z2)
        background.set_grid_descriptor(grid._gd)
        if self.target_potential is None:
            jellium = Jellium(background,
                              natoms=len(relpos_ac),
                              grid=grid)
        else:
            jellium = FixedPotentialJellium(
                background,
                natoms=len(relpos_ac),
                grid=grid,
                workfunction=self.target_potential)
        return SJMEnvironment(solvation, jellium)


class SJMEnvironment(Environment):
    def __init__(self,
                 solvation: Solvation,
                 jellium: Jellium):
        super().__init__(solvation.natoms)
        self.solvation = solvation
        self.jellium = jellium
        self.charge = jellium.charge

    def create_poisson_solver(self, **kwargs):
        ps = self.solvation.create_poisson_solver(**kwargs).solver
        return SJMPoissonSolver(ps, self.solvation.dielectric)

    def post_scf_convergence(self,
                             ibzwfs,
                             nelectrons,
                             occ_calc,
                             mixer,
                             log) -> bool:
        converged = self.jellium.post_scf_convergence(
            ibzwfs, nelectrons, occ_calc, mixer, log)
        self.charge = self.jellium.charge
        return converged

    def update1(self, nt_r):
        self.solvation.update1(nt_r)
        self.jellium.update1(nt_r)

    def update2(self, nt_r, vHt_r, vt_sr) -> float:
        return self.solvation.update2(nt_r, vHt_r, vt_sr)


class SJMPoissonSolver(PoissonSolverWrapper):
    def __init__(self, solver, dielectric):
        super().__init__(solver)
        self.dielectric = dielectric

    def solve(self,
              vHt_r,
              rhot_r) -> float:
        self.solver.solve(vHt_r.data, rhot_r.data)
        eps_r = vHt_r.desc.from_data(self.dielectric.eps_gradeps[0])
        saw_tooth_z = modified_saw_tooth(eps_r)
        s1, s2 = saw_tooth_z[[2, 10]]
        v1, v2 = vHt_r.data[:, :, [2, 10]].mean(axis=(0, 1))
        vHt_r.data -= (v2 - v1) / (s2 - s1) * saw_tooth_z[np.newaxis,
                                                          np.newaxis]
        vHt_r.data -= vHt_r.data[:, :, -1].mean()
        return np.nan


def modified_saw_tooth(eps_r: UGArray) -> np.ndarray:
    a_z = 1.0 / eps_r.data.mean(axis=(0, 1))
    saw_tooth_z = np.add.accumulate(a_z)
    saw_tooth_z -= 0.5 * a_z
    return saw_tooth_z
