import numpy as np
from gpaw.core import PWDesc
from gpaw.core.atom_arrays import AtomDistribution
from gpaw.setup import Setups


class PAWPosissonSolver:
    def __init__(self,
                 pw: PWDesc,
                 setups: Setups,
                 poisson_solver,
                 fracpos_ac: np.ndarray,
                 atomdist: AtomDistribution,
                 xp=np):
        self.xp = xp
        self.ghat_aLh = setups.create_compensation_charges(
            pw, fracpos_ac, atomdist, xp)

    def solve_extra(self, nt_g):
        charge_h = vHt_h.desc.zeros(xp=self.xp)
        coef_aL = density.calculate_compensation_charge_coefficients()
        self.ghat_aLh.add_to(charge_h, coef_aL)

        if pw.comm.rank == 0:
            for rank, g in enumerate(self.g_r):
                if rank == 0:
                    charge_h.data[self.h_g] += nt0_g.data[g]
                else:
                    pw.comm.send(nt0_g.data[g], rank)
        else:
            data = self.xp.empty(len(self.h_g), complex)
            pw.comm.receive(data, 0)
            charge_h.data[self.h_g] += data

        # background charge ???

        e_coulomb = self.poisson_solver.solve(vHt_h, charge_h)
        