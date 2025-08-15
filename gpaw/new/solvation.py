import numpy as np
from ase.units import Ha, Bohr
from gpaw.fd_operators import Gradient
from gpaw.new.c import add_to_density
from gpaw.new.environment import Environment
from gpaw.new.poisson import PoissonSolver, PoissonSolverWrapper
from gpaw.solvation.poisson import WeightedFDPoissonSolver
from gpaw.solvation.cavity import Cavity
from gpaw.solvation.dielectric import Dielectric
from gpaw.solvation.interactions import Interaction
from gpaw.dft import Parameter


class Solvation(Parameter):
    name = 'solvation'

    def __init__(self, cavity, dielectric, interactions=None):
        print(cavity)
        self.cavity = Cavity.from_dict(cavity)
        self.dielectric = Dielectric.from_dict(dielectric)
        self.interactions = [Interaction.from_dict(i)
                             for i in interactions or []]

    def todict(self):
        return {'cavity': self.cavity.todict(),
                'dielectric': self.dielectric.todict(),
                'interactions': [
                    {'name': i.__class__.__name__, **i.todict()}
                    for i in self.interactions]}

    def build(self,
              setups,
              grid,
              relpos_ac,
              log,
              comm):
        return SolvationEnvironment(
            cavity=self.cavity,
            dielectric=self.dielectric,
            interactions=self.interactions,
            setups=setups, grid=grid, relpos_ac=relpos_ac,
            log=log, comm=comm)


class SolvationEnvironment(Environment):
    def __init__(self,
                 *,
                 cavity,
                 dielectric,
                 interactions=None,
                 setups, grid, relpos_ac, log, comm):
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
                           scaled_positions=relpos_ac,
                           cell=grid.cell * Bohr,
                           pbc=grid.pbc)
        self.cavity.update_atoms(self.atoms, log)
        for ia in self.interactions:
            ia.update_atoms(self.atoms, log)
        self.grad_v = [Gradient(grid, v, 1.0, n=3) for v in range(3)]
        self.vt_ia_r = grid.empty()  # self.finegd.zeros()
        self.e_interactions = np.nan
        super().__init__(len(self.atoms))

    def interaction_energy(self):
        return self.e_interactions * Ha

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

    def update2(self, nt_r, vHt_r, vt_sr):
        if self.cavity.depends_on_el_density:
            del_g_del_n_g = self.cavity.del_g_del_n_g
            del_eps_del_g_g = self.dielectric.del_eps_del_g_g
            Veps = -1 / (8 * np.pi) * del_eps_del_g_g * del_g_del_n_g
            Veps *= grad_squared(vHt_r, self.grad_v).data
            for vt_r in vt_sr.data:
                vt_r += Veps

        density = DensityWrapper(nt_r)
        ia_changed = [
            ia.update(
                self.atoms,
                density,
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
        self.e_interactions = Eias.sum()

        self.cavity.communicate_vol_surf(self.comm)
        for E, ia in zip(Eias, self.interactions):
            pass

        self.atoms = None
        return self.e_interactions

    def forces(self, nt_r, vHt_r):
        F_av = np.zeros((self.natoms, 3))
        add_el_force_correction(
            nt_r, vHt_r, self.grad_v, self.cavity, self.dielectric, F_av)

        density = DensityWrapper(nt_r)

        for ia in self.interactions:
            if self.cavity.depends_on_atomic_positions:
                delta_E_delta_g_r = self.grid.from_data(
                    ia.delta_E_delta_g_g)
                for a, F_v in enumerate(F_av):
                    del_g_del_r_vg = self.grid.from_data(
                        self.cavity.get_del_r_vg(a, density))
                    F_v -= delta_E_delta_g_r.integrate(del_g_del_r_vg,
                                                       skip_sum=True)

            if ia.depends_on_atomic_positions:
                for a, F_v in enumerate(F_av):
                    del_E_del_r_vr = self.grid.from_data(
                        ia.get_del_r_vg(a, density))
                    F_v -= del_E_del_r_vr.integrate(skip_sum=True)

        return F_av

    def write_sjm_traces(self, path='sjm_traces', style='z',props=['cavity']):
                         #props=('potential', 'cavity', 'background_charge')):
        """Write traces of quantities in `props` to file on disk; traces will
        be stored within specified path. Default is to save as vertical traces
        (style 'z'), but can also save as cube (specify `style='cube'`)."""
        data = {'cavity': self.cavity.g_g,}
                #'background_charge': self.density.background_charge.mask_g,
                #'potential': (self.hamiltonian.vHt_g * Ha -
                #              self.get_fermi_level())}
        import os
        if not os.path.exists(path) and self.comm.rank == 0:
            os.makedirs(path)
        for prop in props:
            if style == 'z':
                _write_trace_in_z(self.cavity.gd, data[prop], prop + '.txt', path)
            elif style == 'cube':
                _write_property_on_grid(self.cavity.gd, data[prop], self.atoms,
                                        prop + '.cube', path)


def add_el_force_correction(nt_r, vHt_r, grad_v, cavity, dielectric, F_av):
    if not cavity.depends_on_atomic_positions:
        return

    fixed_r = grad_squared(vHt_r, grad_v)  # XXX grad_vHt_g inexact in bmgs
    fixed_r.data *= 1 / (8 * np.pi) * dielectric.del_eps_del_g_g

    density = DensityWrapper(nt_r)

    for a, F_v in enumerate(F_av):
        del_g_del_r_vr = fixed_r.desc.from_data(
            cavity.get_del_r_vg(a, density))
        F_v += fixed_r.integrate(del_g_del_r_vr, skip_sum=True)


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

def _write_property_on_grid(grid, _property, atoms, name, dir):
    """Writes out a property (like electrostatic potential, cavity, or
    background charge) on the grid, as a cube file. `grid` is the
    grid descriptor, typically self.density.finegd. `property` is the property
    to be output, on the same grid."""
    _property = grid.collect(_property, broadcast=True)
    np.save('test',_property)
    print(atoms)
    #import os
    #from ase.io import write
    #write(os.path.join(dir, name), atoms)#, data=prop)

def _write_trace_in_z(grid, _property, name, dir):
    """Writes out a property (like electrostatic potential, cavity, or
    background charge) as a function of the z coordinate only. `grid` is the
    grid descriptor, typically self.density.finegd. `property` is the property
    to be output, on the same grid."""
    _property = grid.collect(_property, broadcast=True)
    property_z = _property.mean(0).mean(0)
    from ase.parallel import paropen
    import os
    with paropen(os.path.join(dir, name), 'w') as f:
        for i, val in enumerate(property_z):
            f.write(f'{(i + 1) * grid.h_cv[2][2] * Bohr:f} {val:1.8f}\n')
