from __future__ import annotations
import numpy as np
from gpaw.new.poisson import PoissonSolver
from gpaw.core import UGArray, UGDesc, PWArray
from ase.units import Ha
from gpaw.new.ibzwfs import IBZWaveFunctions


class Environment:
    """Environment object.

    Used for jellium, solvation, solvated jellium model, ...
    """
    def __init__(self, natoms: int):
        self.natoms = natoms
        self.charge = 0.0

    def create_poisson_solver(self, *, grid, xp, solver) -> PoissonSolver:
        return solver.build(grid=grid, xp=xp)

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        """Allow for environment to "converge"."""
        return True

    def update1(self, nt_r) -> None:
        """Hook called right before solving the Poisson equation."""
        pass

    def update1pw(self, nt_g) -> None:
        """PW-mode hook called right before solving the Poisson equation."""
        pass

    def update2(self, nt_r, vHt_r, vt_sr) -> float:
        """Calculate environment energy."""
        return 0.0

    def forces(self, nt_r, vHt_r):
        return np.zeros((self.natoms, 3))


