from gpaw.new.environment import Environment, FixedPotentialJellium
from gpaw.new.solvation import Solvation
from gpaw.jellium import create_background_charge


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
            self.cavity,
            self.dielectric,
            self.interactions,
            setups, grid, relpos_ac, log, comm, nn)
        jellium = create_background_charge(
            charge=self.excess_electrons,
            z1=self.jelliumregion['bottom'],
            z2=self.jelliumregion['top'])
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
        self.solvation = solvation
        self.fixed_potential = fixed_potential

        poissonsolver={'dipolelayer': 'xy',
                       'zero_vacuum': True},

    def post_scf_convergence(self,
                             ibzwfs,
                             occ_calc,
                             mixer,
                             log) -> bool:
        self.fixed_potential.post_scf_convergence(
            ibzwfs, occ_calc, mixer, log)
s,