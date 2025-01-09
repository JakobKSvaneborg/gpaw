from __future__ import annotations

import numpy as np
from gpaw.core import PWArray, UGArray, UGDesc
from gpaw.core.atom_arrays import AtomArrays
from gpaw.new.calculation import DFTCalculation
from gpaw.new.pw.hybrids import Psi, coulomb, ifft
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.symmetry import SymmetrizationPlan
from gpaw.setup import Setups
from gpaw.utilities import unpack_hermitian


class NonSelfConsistentHSE06:
    exx_fraction = 0.25
    hse_omega = 0.11

    def __init__(self,
                 ibzwfs: PWFDIBZWaveFunctions,
                 grid: UGDesc,
                 setups: Setups,
                 relpos_ac: np.ndarray):
        self.VC_aii = [unpack_hermitian(setup.X_p * self.exx_fraction)
                       for setup in setups]
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.VV_app = [setup.M_pp * self.exx_fraction for setup in setups]

        xp = np
        self.plan = grid.fft_plans(xp=xp)

        nbands = ibzwfs.nbands
        self.psit_K = ibz2bz(ibzwfs, grid, setups, relpos_ac, 0)
        for psit in self.psit_K:
            psit.psit_nR = grid.empty(nbands)
            ifft(psit.psit_nG, psit.psit_nR, self.plan)
            psit.Q_aniL = {a: np.einsum('ijL, nj -> niL',
                                        setup.Delta_iiL, psit.P_ani[a])
                           for a, setup in enumerate(setups)}

    @classmethod
    def from_dft_calculation(cls,
                             dft: DFTCalculation) -> NonSelfConsistentHSE06:
        return cls(dft.ibzwfs,
                   dft.density.nt_sR.desc,
                   dft.setups,
                   dft.relpos_ac)

    def calculate(self,
                  wfs: PWFDWaveFunctions,
                  n1=0,
                  n2=None) -> np.ndarray:
        n2 = n2 or wfs.nbands
        eig_n = np.zeros(n2 - n1)
        for psit in self.psit_K:
            v_G = coulomb()
            for n, psit_R in enumerate(psit.psit_nR):
                eig_n += self._calculate(v_G,
                                         psit_R, n,
                                         wfs.psit_nX, wfs.P_ani,
                                         n1, n2)
        return eig_n

    def _calculate(self,
                   psit1_R: UGArray,
                   Q1_aniL,
                   n1: int,
                   psit2_nG: PWArray,
                   P2_ani: AtomArrays,
                   n2a: int,
                   n2b: int) -> np.ndarray:
        """
        for n2, (psit2_R, out_G) in enumerate(zips(psi2.psit_nR, Htpsit_nG)):
            rhot_nR.data[:] = psit1_nR.data * psit2_R.data.conj()
            for a, Q1_niL in Q1_aniL.items():
                P2_i = psi2.P_ani[a][n2]
                Q_anL[a][:] = P2_i.conj() @ Q1_niL
        fft(rhot_nR, rhot_nG, plan=self.plan)
        mmm(1.0 / self.pw.dv, Q_anL.data, 'N', self.ghat_GA, 'T',
            1.0, rhot_nG.data)

        e = 0.0
        for n1, (rhot_R, rhot_G, f1) in enumerate(zips(rhot_nR,
                                                       rhot_nG,
                                                       psi1.f_n)):
            vrhot_G.data = rhot_G.data * self.v_G.data
            if psi2.f_n is not None:
                e += f1 * psi2.f_n[n2] * rhot_G.integrate(vrhot_G).real
        """
        print('.')
        return 0.0


def ibz2bz(ibzwfs, grid, setups, relpos_ac, spin):
    ibz = ibzwfs.ibz
    symmetries = ibzwfs.ibz.symmetries
    symmplan = SymmetrizationPlan(symmetries, [setup.l_j for setup in setups])
    psit_K = []
    for wfs_s in ibzwfs.wfs_qs:
        wfs = wfs_s[spin]
        for K, k in enumerate(ibz.bz2ibz_K):
            if k == wfs.k:
                s = ibz.s_K[K]
                U_cc = symmetries.rotation_scc[s]
                complex_conjugate = ibz.time_reversal_K[K]
                psit1_nG = wfs.psit_nX
                psit2_nG = psit1_nG.transform(U_cc, complex_conjugate)
                P1_ani = wfs.P_ani
                P2_ani = P1_ani.new()
                for a, P2_ni in P2_ani.items():
                    b = symmetries.atommap_sa[s, a]
                    S_c = relpos_ac[a] @ U_cc - relpos_ac[b]
                    x = np.exp(2j * np.pi * (psit1_nG.desc.kpt_c @ S_c))
                    U_ii = symmplan.rotations(setups[a].l_j)[s].T * x
                    P2_ni[:] = P1_ani[b] @ U_ii
                if complex_conjugate:
                    np.conjugate(P2_ani.data, P2_ani.data)
                psit = Psi(psit2_nG, P2_ani, wfs.myocc_n)
                psit_K.append(psit)
    return psit_K
