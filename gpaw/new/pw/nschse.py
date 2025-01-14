from __future__ import annotations

import numpy as np
from gpaw.core import PWArray, UGArray, UGDesc
from gpaw.core.atom_arrays import AtomArrays
from gpaw.new import zips as zip
from gpaw.new.calculation import DFTCalculation
from gpaw.new.pw.hybrids import Psi, fft, hse_coulomb, ifft, pawexxvv
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.symmetry import SymmetrizationPlan
from gpaw.setup import Setups
from gpaw.utilities import unpack_hermitian


class NonSelfConsistentHSE06:
    exx_fraction = 0.25
    hse06_omega = 0.11

    def __init__(self,
                 ibzwfs: PWFDIBZWaveFunctions,
                 grid: UGDesc,
                 D_asii: AtomArrays,
                 setups: Setups,
                 relpos_ac: np.ndarray):
        self.grid = grid
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.dE_asii = []
        for D_sii, setup in zip(D_asii.values(), setups):
            VC_ii = unpack_hermitian(setup.X_p * self.exx_fraction)
            self.dE_asii.append(
                [self.exx_fraction * 4 *
                 (pawexxvv(2 * setup.M_pp, D_ii / ibzwfs.spin_degeneracy) +
                  VC_ii)
                 for D_ii in D_sii])

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

        self.ghat_aLR = setups.create_compensation_charges(grid, relpos_ac)

    @classmethod
    def from_dft_calculation(cls,
                             dft: DFTCalculation) -> NonSelfConsistentHSE06:
        assert isinstance(dft.ibzwfs, PWFDIBZWaveFunctions)
        return cls(dft.ibzwfs,
                   dft.density.nt_sR.desc.new(dtype=complex),
                   dft.density.D_asii,
                   dft.setups,
                   dft.relpos_ac)

    def calculate(self,
                  wfs: PWFDWaveFunctions,
                  na: int = 0,
                  nb: int | None = None) -> np.ndarray:
        n2a = na
        n2b = nb or wfs.nbands
        P2_ani = {a: P_ni[n2a:n2b] for a, P_ni in wfs.P_ani.items()}
        ut2_nR = self.grid.empty(n2b - n2a)
        psit2_nG = wfs.psit_nX[n2a:n2b]
        pw2 = psit2_nG.desc
        ifft(psit2_nG, ut2_nR, self.plan)
        eig_n = np.zeros(n2b - n2a)
        for psit1 in self.psit_K:
            pw1 = psit1.psit_nG.desc
            pw = pw1.new(kpt=pw1.kpt_c - pw2.kpt_c)
            v_G = hse_coulomb(pw, self.hse06_omega)
            for n1, psit1_R in enumerate(psit1.psit_nR):
                eig_n += self._calculate(v_G,
                                         psit1, n1,
                                         ut2_nR,
                                         P2_ani)
        eig_n /= len(self.psit_K)

        # Valence-valence and valence-core PAW corrections:
        for P2_ni, dE_sii in zip(P2_ani.values(), self.dE_asii):
            eig_n -= np.einsum('ni, ij, nj -> n',
                               P2_ni.conj(), dE_sii[wfs.spin], P2_ni).real
            print('de', np.einsum('ni, ij, nj -> n',
                                  P2_ni.conj(), dE_sii[wfs.spin], P2_ni).real)
        return eig_n

    def _calculate(self,
                   v_G: PWArray,
                   psit1: Psi,
                   n1: int,
                   ut2_nR: UGArray,
                   P2_ani: dict[int, np.ndarray]) -> np.ndarray:
        rhot_nR = ut2_nR.copy()
        ut1_nR = psit1.psit_nR
        assert ut1_nR is not None
        rhot_nR.data *= ut1_nR.data[n1].conj()
        Q_aniL = psit1.Q_aniL
        assert Q_aniL is not None
        Q_anL = {}
        for a, Q1_niL in Q_aniL.items():
            Q_anL[a] = P2_ani[a].conj() @ Q1_niL[n1]
        self.ghat_aLR.add_to(rhot_nR, Q_anL)
        rhot_nG = v_G.desc.empty(len(rhot_nR))
        fft(rhot_nR, rhot_nG, plan=self.plan)
        rhot_nG.data *= v_G.data.real**0.5
        e_n = rhot_nG.norm2()
        print(n1, v_G.desc.kpt, e_n)
        return e_n


def ibz2bz(ibzwfs: PWFDIBZWaveFunctions,
           grid: UGDesc,
           setups: Setups,
           relpos_ac: np.ndarray,
           spin: int) -> list[Psi]:
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
