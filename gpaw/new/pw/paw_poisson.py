"""PAW Poisson-solvers.

See equations (25-28) in
P. E. Blöchl: https://sci-hub.st/10.1103/PhysRevB.50.17953
"""
from __future__ import annotations

from math import pi

import numpy as np
from ase.neighborlist import primitive_neighbor_list
from scipy.special import erf

from gpaw.atom.radialgd import EquidistantRadialGridDescriptor as RGD
from gpaw.atom.shapefunc import shape_functions
from gpaw.core import PWArray, PWDesc
from gpaw.core.atom_arrays import AtomArrays, AtomDistribution
from gpaw.gpu import cupy as cp
from gpaw.setup import Setups
from gpaw.spline import Spline
from gpaw.lcao.overlap import (FourierTransformer, TwoSiteOverlapCalculator,
                               ManySiteOverlapCalculator)


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
    f *= 16 / pi / rc1**3 / rc2**3
    if r == 0.0:
        return f
    T = a1 * a2 / (a1 + a2) * r**2
    y = 0.5 * f * erf(T**0.5) * (pi / T)**0.5
    return y


def dcdr(r, rc1, rc2):
    if r == 0.0:
        return 0.0
    a1 = 1 / rc1**2
    a2 = 1 / rc2**2
    f = 2 * (pi**5 / (a1 + a2))**0.5 / (a1 * a2)
    f *= 16 / pi / rc1**3 / rc2**3
    T = a1 * a2 / (a1 + a2) * r**2
    y = 0.5 * f * erf(T**0.5) * (pi / T)**0.5
    dydr = (2 / pi**0.5 * np.exp(-T) - y) / r
    return dydr


def tci(rcut, I_a, dghat_Il, vhat_Il):
    transformer = FourierTransformer(rcut=rcut, N=2**8)
    tsoc = TwoSiteOverlapCalculator(transformer)
    msoc = ManySiteOverlapCalculator(tsoc, I_a, I_a)
    dghat_Ilq = msoc.transform(dghat_Il)
    vhat_Ilq = msoc.transform(vhat_Il)
    l_Il = [[dghat.l for dghat in dghat_l] for dghat_l in dghat_Il]
    expansions = msoc.calculate_expansions(l_Il, dghat_Ilq,
                                           l_Il, vhat_Ilq)
    return expansions


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
        self.fracpos_ac = fracpos_ac
        self.cutoff_a = np.asarray(cutoff_a)
        self.r2 = self.cutoff_a.max() * 2
        self.rcut = 5 * self.r2
        d = 0.01
        rgd = RGD(d, int(self.rcut / d))
        g0_lg = shape_functions(rgd, 'gauss', self.r2, lmax=2)
        ghat_l = [rgd.spline(g_g, l=l) for l, g_g in enumerate(g0_lg)]
        ghat_al = [ghat_l] * len(self.cutoff_a)
        cache: dict[float, list[Spline]] = {}
        dghat_Il = []
        vhat_Il = []
        vhat_al = []
        I_a = []
        for r1 in cutoff_a:
            if r1 in cache:
                I, dghat_l, vhat_l = cache[r1]
            else:
                g_lg = shape_functions(rgd, 'gauss', r1, lmax=2)
                dghat_l = [rgd.spline(g_g - g0_lg[l], l=l)
                           for l, g_g in enumerate(g_lg)]
                v_lg = dvl(rgd, r1, self.r2, lmax=2)
                vhat_l = [rgd.spline(v_g * (4 * pi), l=l)
                          for l, v_g in enumerate(v_lg)]
                I = len(cache)
                cache[r1] = I, dghat_l, vhat_l
                dghat_Il.append(dghat_l)
                vhat_Il.append(vhat_l)
            I_a.append(I)
            vhat_al.append(vhat_l)

        self.ghat_aLg = pwg.atom_centered_functions(
            ghat_al, fracpos_ac, atomdist=atomdist, xp=xp)
        self.vhat_aLg = pwg.atom_centered_functions(
            vhat_al, fracpos_ac, atomdist=atomdist, xp=xp)

        self.expansion = tci(self.rcut, I_a, dghat_Il, vhat_Il)
        print(self.expansion)

        self._neighbors = None
        self.ghat_aLh = self.ghat_aLg  # old name

    def get_neighbors(self):
        if self._neighbors is None:
            pw = self.pwg
            self._neighbors = primitive_neighbor_list(
                'ijdD', pw.pbc, pw.cell, self.fracpos_ac,
                2 * self.rcut,
                use_scaled_positions=True,
                self_interaction=True)
        return self._neighbors

    def dipole_layer_correction(self):
        return self.poisson_solver.dipole_layer_correction()

    def move(self, fracpos_ac, atomdist):
        self.fracpos_ac = fracpos_ac
        self.ghat_aLg.move(fracpos_ac, atomdist)
        self.vhat_aLg.move(fracpos_ac, atomdist)
        self._neighbors = None

    def solve(self,
              nt_g: PWArray,
              Q_aL: AtomArrays,
              vt0_g: PWArray,
              vHt_g: PWArray | None = None):
        charge_g = nt_g.copy()
        self.ghat_aLg.add_to(charge_g, Q_aL)
        pwg = self.pwg

        if vHt_g is None:
            vHt_g = pwg.empty(xp=self.xp)

        e_coulomb1 = self.poisson_solver.solve(vHt_g, charge_g)

        vhat_g = pwg.empty()  # MYPY
        vhat_g.data[:] = 0.0  # MYPY

        self.vhat_aLg.add_to(vhat_g, Q_aL)
        vt0_g.data += vhat_g.data
        e_coulomb2 = vhat_g.integrate(nt_g)

        V_aL = self.ghat_aLg.integrate(vHt_g)
        self.vhat_aLg.integrate(nt_g, V_aL, add_to=True)

        e_coulomb3 = 0.0
        for a1, a2, d, d_v in zip(*self.get_neighbors()):
            v = Q_aL[a2][0] * (
                c(d, self.r2, self.r2) -
                c(d, self.cutoff_a[a1], self.cutoff_a[a2])) / 4 / pi
            V_aL[a1][0] -= v
            e_coulomb3 += Q_aL[a1][0] * v
        e_coulomb3 *= -0.5

        vHt0_g = vHt_g.gather()
        if pwg.comm.rank == 0:
            vt0_g.data += vHt0_g.data

        return e_coulomb1 + e_coulomb2 + e_coulomb3, vHt_g, V_aL

    def force_contribution(self, Q_aL):
        force_av = np.zeros((len(Q_aL), 3))
        for a1, a2, d, d_v in zip(*self.get_neighbors()):
            v = Q_aL[a1][0] * Q_aL[a2][0] * (
                dcdr(d, self.r2, self.r2) -
                dcdr(d, self.cutoff_a[a1], self.cutoff_a[a2])) / 4 / pi
            if d > 0:
                f_v = v * d_v / d
                force_av[a1] += f_v
                force_av[a2] -= f_v


class SimplePAWPoissonSolver:
    def __init__(self,
                 pwg: PWDesc,
                 cutoff_a,
                 poisson_solver,
                 fracpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np):
        self.xp = xp
        self.pwg = pwg
        self.pwg0 = pwg.new(comm=None)  # not distributed
        self.poisson_solver = poisson_solver
        d = 0.005
        rgd = RGD(d, int(10.0 / d))
        cache: dict[float, list[Spline]] = {}
        ghat_al = []
        for rc in cutoff_a:
            if rc in cache:
                ghat_l = cache[rc]
            else:
                g_lg = shape_functions(rgd, 'gauss', rc, lmax=2)
                ghat_l = [rgd.spline(g_g, l=l) for l, g_g in enumerate(g_lg)]
                cache[rc] = ghat_l
            ghat_al.append(ghat_l)

        self.ghat_aLg = pwg.atom_centered_functions(
            ghat_al, fracpos_ac, atomdist=atomdist, xp=xp)

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
            vHt_g = pwg.empty(xp=self.xp)
        e_coulomb = self.poisson_solver.solve(vHt_g, charge_g)
        vHt0_g = vHt_g.gather()
        if pwg.comm.rank == 0:
            vt0_g.data += vHt0_g.data
        V_aL = self.ghat_aLg.integrate(vHt_g)
        return e_coulomb, vHt_g, V_aL


class OldPAWPoissonSolver:
    def __init__(self,
                 pwg: PWDesc,
                 # cutoff_a,
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

    def move(self, fracpos_ac, atomdist):
        self.ghat_aLh.move(fracpos_ac, atomdist)

    def solve(self, nt_g, Q_aL, vt0_g, vHt_h=None):
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
        # print('OLD', e_coulomb)
        # print(charge_h.data[:5])
        # print(vHt_h.data[:5])

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    r = np.linspace(0.01, 10, 101)
    plt.plot(r, c(r, 1, 1) - c(r, 0.5, 0.5))
    plt.show()
