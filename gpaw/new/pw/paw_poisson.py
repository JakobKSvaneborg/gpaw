import numpy as np
from gpaw.core import PWDesc
from gpaw.core.atom_arrays import AtomDistribution
from gpaw.gpu import cupy as cp
from gpaw.setup import Setups


class PAWPosissonSolver:
    def __init__(self,
                 pwg: PWDesc,
                 pwg0: PWDesc,
                 setups: Setups,
                 poisson_solver,
                 fracpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np):
        self.xp = xp
        pwh = poisson_solver.pw
        self.ghat_aLh = setups.create_compensation_charges(
            pwh, fracpos_ac, atomdist, xp)
        self.h_g, self.g_r = pwh.map_indices(self.pw0)
        if xp is cp:
            self.h_g = cp.asarray(self.h_g)
            self.g_r = [cp.asarray(g) for g in self.g_r]

    def solve(self, nt_g, Q_aL, vHt_h):
        charge_h = vHt_h.desc.zeros(xp=self.xp)
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

        e_coulomb = self.poisson_solver.solve(vHt_h, charge_h)

        if pwg.comm.rank == 0:
            vt0_g = self.vbar0_g.copy()
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    vt0_g.data[g] += vHt_h.data[self.h_g]
                else:
                    data = self.xp.empty(len(g), complex)
                    pwg.comm.receive(data, rank)
                    vt0_g.data[g] += data
        else:
            pwg.comm.send(vHt_h.data[self.h_g], 0)

        return e_coulomb
