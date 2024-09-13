from math import pi

import numpy as np
from scipy.special import erf
from ase.neighborlist import primitive_neighbor_list
from gpaw.core import PWDesc, PWArray
from gpaw.core.atom_arrays import AtomDistribution, AtomArrays
from gpaw.gpu import cupy as cp
from gpaw.setup import Setups
from gpaw.atom.shapefunc import shape_functions
from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD


def dv(r, rc, rcut):
    return erf(r / rc) / r - erf(r / rcut) / r


def dvl(rgd, rc, rcut, lmax=2):
    r_g = rgd.r_g.copy()
    r_g[0] = 1.0
    v_g = dv(r_g, rc, rcut)
    v_g[0] = (4.0 / np.pi)**0.5 * (1.0 / rc - 1.0 / rcut)
    return [v_g, v_g * 0.0, v_g * 0.0]


def c(r, rc1, rc2):
    a1 = 1 / rc1**2
    a2 = 1 / rc2**2
    f = 2 * (pi**5 / (a1 + a2))**0.5 / (a1 * a2)
    T = a1 * a2 / (a1 + a2) * r**2
    return 0.5 * f * erf(T**0.5) * (pi / T)**0.5


class PAWPoissonSolver:
    def __init__(self,
                 pwg: PWDesc,
                 cutoff_a: np.ndarray,
                 poisson_solver,
                 fracpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np):
        self.xp = xp
        self.pwg = pwg
        self.pwg0 = pwg.new(comm=None)  # not distributed
        self.poisson_solver = poisson_solver
        cutoff_a = np.asarray(cutoff_a)
        rcut = cutoff_a.max() * 2
        d = 0.01
        rgd = RGD(d, int(rcut * 5 / d))
        g_lg = shape_functions(rgd, 'gauss', rcut, lmax=2)
        ghat_l = [rgd.spline(g_g, l=1) for l, g_g in enumerate(g_lg)]
        ghat_al = [ghat_l] * len(cutoff_a)
        cache = {}
        vhat_al = []
        for rc in cutoff_a:
            if rc in cache:
                vhat_l = cache[rc]
            else:
                v_lg = dvl(rgd, rc, rcut, lmax=2)
                vhat_l = [rgd.spline(v_g, l=1)
                          for l, v_g in enumerate(v_lg)]
                cache[rc] = vhat_l
            vhat_al.append(vhat_l)

        self.ghat_aLg = pwg.atom_centered_functions(
            ghat_al, fracpos_ac, atomdist=atomdist, xp=xp)
        self.vhat_aLg = pwg.atom_centered_functions(
            vhat_al, fracpos_ac, atomdist=atomdist, xp=xp)

        self._neighbors = None

    def get_neighbors(self):
        if self._neighbors is None:
            self._neighbors = primitive_neighbor_list(
                'ijdD', )
        return self._neighbors

    def dipole_layer_correction(self):
        return self.poisson_solver.dipole_layer_correction()

    def solve(self,
              nt_g: PWArray,
              Q_aL: AtomArrays,
              vt0_g: PWArray,
              vHt_g: PWArray | None = None):
        charge_g = nt_g.copy()
        self.ghat_aLg.add_to(charge_g, Q_aL)
        pwg = self.pwg

        if vHt_g is None:
            vHt_g = pwg.zeros(xp=self.xp)

        e_coulomb1 = self.poisson_solver.solve(vHt_g, charge_g)

        vhat_g = pwg.zeros()
        self.vhat_aLg.add_to(vhat_g, Q_aL)
        vHt_g.data += vhat_g.data
        e_coulomb2 = vhat_g.integrate(nt_g)

        e_coulomb3 = 0.0
        for i, j, d, D in self.get_neighbors():
            print(i, j, d, D)

        vHt0_g = vHt_g.gather()
        if pwg.comm.rank == 0:
            vt0_g.data += vHt0_g.data

        V_aL = self.ghat_aLg.integrate(vHt_g)

        return e_coulomb1 + e_coulomb2 + e_coulomb3, vHt_g, V_aL


class SimplePAWPoissonSolver:
    def __init__(self,
                 pwg: PWDesc,
                 setups: Setups,
                 poisson_solver,
                 fracpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np):
        self.xp = xp
        self.pwg = pwg
        self.pwg0 = pwg.new(comm=None)  # not distributed
        self.poisson_solver = poisson_solver
        self.ghat_aLg = setups.create_compensation_charges(
            pwg, fracpos_ac, atomdist, xp)

    def dipole_layer_correction(self):
        return self.poisson_solver.dipole_layer_correction()

    def solve(self,
              nt_g: PWArray,
              Q_aL: AtomArrays,
              vt0_g: PWArray,
              vHt_g: PWArray | None = None):
        charge_g = nt_g.copy()
        self.ghat_aLg.add_to(charge_g, Q_aL)
        pwg = self.pwg
        if vHt_g is None:
            vHt_g = pwg.zeros(xp=self.xp)
        e_coulomb = self.poisson_solver.solve(vHt_g, charge_g)
        vHt0_g = vHt_g.gather()
        if pwg.comm.rank == 0:
            vt0_g.data += vHt0_g.data
        V_aL = self.ghat_aLg.integrate(vHt_g)
        return e_coulomb, vHt_g, V_aL


class OldPAWPoissonSolver:
    def __init__(self,
                 pwg: PWDesc,
                 setups: Setups,
                 poisson_solver,
                 fracpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np):
        self.xp = xp
        self.pwg = pwg
        self.pwg0 = pwg.new(comm=None)  # not distributed
        self.pwh = poisson_solver.pw
        self.poisson_solver = poisson_solver
        self.ghat_aLh = setups.create_compensation_charges(
            self.pwh, fracpos_ac, atomdist, xp)
        self.h_g, self.g_r = self.pwh.map_indices(self.pwg0)
        if xp is cp:
            self.h_g = cp.asarray(self.h_g)
            self.g_r = [cp.asarray(g) for g in self.g_r]

    def dipole_layer_correction(self):
        return self.poisson_solver.dipole_layer_correction()

    def solve(self, nt_g, Q_aL, vt0_g, vHt_h):
        charge_h = self.pwh.zeros(xp=self.xp)
        self.ghat_aLh.add_to(charge_h, Q_aL)
        pwg = self.pwg

        if pwg.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    charge_h.data[self.h_g] += nt_g.data[g]
                else:
                    pwg.comm.send(nt_g.data[g], rank)
        else:
            data = self.xp.empty(len(self.h_g), complex)
            pwg.comm.receive(data, 0)
            charge_h.data[self.h_g] += data

        if vHt_h is None:
            vHt_h = self.pwh.zeros(xp=self.xp)

        e_coulomb = self.poisson_solver.solve(vHt_h, charge_h)

        if pwg.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    vt0_g.data[g] += vHt_h.data[self.h_g]
                else:
                    data = self.xp.empty(len(g), complex)
                    pwg.comm.receive(data, rank)
                    vt0_g.data[g] += data
        else:
            pwg.comm.send(vHt_h.data[self.h_g], 0)

        V_aL = self.ghat_aLh.integrate(vHt_h)

        return e_coulomb, vHt_h, V_aL
