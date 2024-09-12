import numpy as np
from gpaw.core import PWDesc, PWArray
from gpaw.core.atom_arrays import AtomDistribution, AtomArrays
from gpaw.gpu import cupy as cp
from gpaw.setup import Setups


class PAWPoissonSolver:
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
              vHt_g: PWArray):
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
