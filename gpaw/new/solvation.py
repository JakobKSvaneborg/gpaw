import numpy as np

from gpaw.new.environment import Environment
from gpaw.solvation.poisson import WeightedFDPoissonSolver
from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.fd_operators import Gradient
from gpaw.new.c import add_to_density


class Solvation(Environment):
    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions=None,
                 setups, grid, fracpos_ac, log, comm, nn):
        self.cavity = cavity
        self.dielectric = dielectric
        self.interactions = interactions or []
        finegd = grid._gd
        self.grid = grid
        self.comm = comm
        self.cavity.set_grid_descriptor(finegd)
        self.dielectric.set_grid_descriptor(finegd)
        for ia in self.interactions:
            ia.set_grid_descriptor(finegd)
        self.cavity.allocate()
        self.dielectric.allocate()
        for ia in self.interactions:
            ia.allocate()
        from ase import Atoms
        self.atoms = Atoms([setup.symbol for setup in setups],
                           scaled_positions=fracpos_ac,
                           cell=grid.cell,
                           pbc=grid.pbc)
        self.cavity.update_atoms(self.atoms, log)
        for ia in self.interactions:
            ia.update_atoms(self.atoms, log)
        self.grad_v = [Gradient(grid, v, 1.0, nn) for v in range(3)]
        self.vt_ia_r = grid.empty()  # self.finegd.zeros()

    def create_poisson_solver(self, grid, *, xp, **kwargs) -> PoissonSolver:
        psolver = WeightedFDPoissonSolver()
        psolver.set_dielectric(self.dielectric)
        psolver.set_grid_descriptor(self.grid._gd)
        return PoissonSolverWrapper(psolver)

    def update1(self, nt_r, kin_en_using_band=True):
        density = DensityWrapper(nt_r)
        self.cavity_changed = self.cavity.update(self.atoms, density)
        if self.cavity_changed:
            self.cavity.update_vol_surf()
            self.dielectric.update(self.cavity)

    def update2(self, vHt_r, vt_sr):
        if self.cavity.depends_on_el_density:
            del_g_del_n_g = self.cavity.del_g_del_n_g
            del_eps_del_g_g = self.dielectric.del_eps_del_g_g
            Veps = -1 / (8 * np.pi) * del_eps_del_g_g * del_g_del_n_g
            Veps *= grad_squared(vHt_r, self.grad_v).data
            for vt_r in vt_sr.data:
                vt_r += Veps

        ia_changed = [
            ia.update(
                self.atoms,
                None,  # density,
                self.cavity if self.cavity_changed else None)
            for ia in self.interactions]
        if any(ia_changed):
            self.vt_ia_r.data.fill(.0)
            for ia in self.interactions:
                if ia.depends_on_el_density:
                    self.vt_ia_r.data += ia.delta_E_delta_n_g
                if self.cavity.depends_on_el_density:
                    self.vt_ia_r.data += (ia.delta_E_delta_g_g *
                                          self.cavity.del_g_del_n_g)
        if len(self.interactions) > 0:
            for vt_r in vt_sr.data:
                vt_r += self.vt_ia_r.data
        Eias = np.array([ia.E for ia in self.interactions])
        self.grid.comm.sum(Eias)

        self.cavity.communicate_vol_surf(self.comm)
        for E, ia in zip(Eias, self.interactions):
            pass
            # setattr(self, 'e_' + ia.subscript, E)

        self.atoms = None


class DensityWrapper:
    def __init__(self, nt_r):
        self.nt_g = nt_r.data


def grad_squared(a_r, grad_v):
    tmp_r = a_r.new()
    b_r = a_r.desc.zeros()
    for grad in grad_v:
        grad(a_r, tmp_r)
        add_to_density(1, tmp_r.data, b_r.data)
    return b_r
