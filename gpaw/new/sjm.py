from ase.units import Bohr
from gpaw.new.environment import Environment, FixedPotentialJellium
from gpaw.new.solvation import Solvation
from gpaw.jellium import create_background_charge
from gpaw.poisson import DipoleCorrection
from gpaw.new.poisson import PoissonSolverWrapper


class SJM:
    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions,
                 jelliumregion,
                 target_potential: float,
                 excess_electrons: float = 0.0,
                 tol: float = 0.001):
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions
        self.jelliumregion = jelliumregion
        self.target_potential = target_potential
        self.excess_electrons = excess_electrons
        self.tol = tol

    def build(self, setups, grid, relpos_ac, log, comm, nn):
        solvation = Solvation(
            cavity=self.cavity,
            dielectric=self.dielectric,
            interactions=self.interactions,
            setups=setups, grid=grid, relpos_ac=relpos_ac,
            log=log, comm=comm, nn=nn)
        h = grid.cell_cv[2, 2] * Bohr
        z1 = relpos_ac[:, 2].max() * h + 3.0
        z2 = self.jelliumregion.get('top', h - 1.0)
        jellium = create_background_charge(charge=self.excess_electrons,
                                           z1=z1,
                                           z2=z2)
        jellium.set_grid_descriptor(grid._gd)
        fixed_pot = FixedPotentialJellium(
            jellium,
            natoms=len(relpos_ac),
            grid=grid,
            workfunction=self.target_potential)
        return SJMEnvironment(solvation, fixed_pot)


class SJMEnvironment(Environment):
    def __init__(self,
                 solvation: Solvation,
                 fixed_potential: FixedPotentialJellium):
        super().__init__(solvation.natoms)
        self.solvation = solvation
        self.fixed_potential = fixed_potential
        self.charge = self.fixed_potential.charge

    def create_poisson_solver(self, **kwargs):
        ps = self.solvation.create_poisson_solver(**kwargs).solver
        return PoissonSolverWrapper(
            DipoleCorrection(ps, direction=2, zero_vacuum=True))

    def post_scf_convergence(self,
                             ibzwfs,
                             occ_calc,
                             mixer,
                             log) -> bool:
        return self.fixed_potential.post_scf_convergence(
            ibzwfs, occ_calc, mixer, log)

    def update1(self, nt_r):
        self.solvation.update1(nt_r)
        self.fixed_potential.update1(nt_r)

    def update2(self, nt_r, vHt_r, vt_sr):
        return self.solvation.update2(nt_r, vHt_r, vt_sr)
