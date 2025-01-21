from __future__ import annotations

import numpy as np
from ase.units import Ha

from gpaw.core import PWArray, UGArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.new import zips as zip
from gpaw.new.c import add_to_density
from gpaw.new.calculation import DFTCalculation
from gpaw.new.density import Density
from gpaw.new.pw.hybrids import Psi, fft, pawexxvv, truncated_coulomb
from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.xc import create_functional
from gpaw.setup import Setups
from gpaw.utilities import pack_density, unpack_hermitian


class NonSelfConsistentHSE06:
    exx_fraction = 0.25
    hse06_omega = 0.11

    @classmethod
    def from_dft_calculation(cls,
                             dft: DFTCalculation) -> NonSelfConsistentHSE06:
        assert isinstance(dft.ibzwfs, PWFDIBZWaveFunctions)
        return cls(dft.ibzwfs,
                   dft.density,
                   dft.pot_calc,
                   dft.setups,
                   dft.relpos_ac)

    def __init__(self,
                 ibzwfs: PWFDIBZWaveFunctions,
                 density: Density,
                 pot_calc: PlaneWavePotentialCalculator,
                 setups: Setups,
                 relpos_ac: np.ndarray):
        self.grid = density.nt_sR.desc.new(dtype=complex)
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]

        xp = np
        self.plan = self.grid.fft_plans(xp=xp)

        # ???? Spin ????
        nocc, self.psit_K = ibz2bz(ibzwfs, setups, relpos_ac, 0)
        for psit in self.psit_K.values():
            psit.psit_nR = self.grid.empty(nocc)
            psit.psit_nG.ifft(out=psit.psit_nR, plan=self.plan, periodic=False)
            psit.Q_aniL = {a: np.einsum('ijL, nj -> niL',
                                        setup.Delta_iiL, psit.P_ani[a].conj())
                           for a, setup in enumerate(setups)}

        self.ghat_aLR = setups.create_compensation_charges(
            self.grid, relpos_ac)
        self.relpos_ac = relpos_ac
        self.setups = setups

        self.dvxct_sR, dVxc_asii = nsc_corrections(density, pot_calc)

        self.dE_asii = []
        for D_sii, setup, dVxc_sii in zip(density.D_asii.values(),
                                          setups,
                                          dVxc_asii.values()):
            VC_ii = unpack_hermitian(setup.X_p * self.exx_fraction)
            dE_sii = []
            for D_ii, dVxc_ii in zip(D_sii, dVxc_sii):
                VV_ii = self.exx_fraction * (
                    pawexxvv(2 * setup.M_pp, D_ii / ibzwfs.spin_degeneracy))
                dE_ii = dVxc_ii - VC_ii - VV_ii
                dE_sii.append(dE_ii)
            self.dE_asii.append(dE_sii)

    def calculate(self,
                  wfs: PWFDWaveFunctions,
                  na: int = 0,
                  nb: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues (in eV)."""
        n2a = na
        n2b = nb or wfs.nbands
        P2_ani = {a: P_ni[n2a:n2b] for a, P_ni in wfs.P_ani.items()}
        ut2_nR = self.grid.empty(n2b - n2a)
        psit2_nG = wfs.psit_nX[n2a:n2b]
        psit2_nG.ifft(out=ut2_nR, plan=self.plan, periodic=False)

        deig_n = self._semi_local_xc_part(ut2_nR, wfs.spin)

        pw2 = psit2_nG.desc
        eig_n = np.zeros(n2b - n2a)
        for psit1 in self.psit_K.values():
            pw1 = psit1.psit_nG.desc
            pw = pw1.new(kpt=pw2.kpt_c - pw1.kpt_c)
            v_G = truncated_coulomb(pw, self.hse06_omega)
            assert psit1.psit_nR is not None
            for n1, psit1_R in enumerate(psit1.psit_nR):
                eig_n += self._exx_part(v_G,
                                        psit1, n1,
                                        ut2_nR,
                                        P2_ani)
        eig_n *= -self.exx_fraction / len(self.psit_K)

        # PAW corrections:
        for P2_ni, dE_sii in zip(P2_ani.values(), self.dE_asii):
            eig_n += np.einsum('ni, ij, nj -> n',
                               P2_ni.conj(), dE_sii[wfs.spin], P2_ni).real

        eig0_n = wfs.eig_n[n2a:n2b]

        return eig0_n * Ha, (deig_n + eig_n + eig0_n) * Ha

    def _exx_part(self,
                  v_G: PWArray,
                  psit1: Psi,
                  n1: int,
                  ut2_nR: UGArray,
                  P2_ani: dict[int, np.ndarray]) -> np.ndarray:
        """"""
        ut1_nR = psit1.psit_nR
        Q1_aniL = psit1.Q_aniL
        f1_n = psit1.f_n
        assert ut1_nR is not None
        assert Q1_aniL is not None
        assert f1_n is not None
        rhot_nR = ut2_nR.copy()
        rhot_nR.data *= ut1_nR.data[n1].conj()
        phase_a = np.exp(-2j * np.pi * (self.relpos_ac @ v_G.desc.kpt_c))
        Q_anL = {}
        if 1:
            for a, Q1_niL in Q1_aniL.items():
                Q_anL[a] = P2_ani[a] @ Q1_niL[n1]
            ghat_aLG = self.setups.create_compensation_charges(
                v_G.desc, self.relpos_ac)
            rhot_nG = v_G.desc.empty(len(rhot_nR))
            fft(rhot_nR, rhot_nG, plan=self.plan)
            ghat_aLG.add_to(rhot_nG, Q_anL)
        else:
            for a, Q1_niL in Q1_aniL.items():
                Q_anL[a] = P2_ani[a] @ Q1_niL[n1] * phase_a[a]
            self.ghat_aLR.add_to(rhot_nR, Q_anL)
            rhot_nG = v_G.desc.empty(len(rhot_nR))
            fft(rhot_nR, rhot_nG, plan=self.plan)
        rhot_nG.data *= v_G.data.real**0.5
        e_n = rhot_nG.norm2()
        return e_n * f1_n[n1]

    def _semi_local_xc_part(self,
                            ut2_nR: UGArray,
                            spin: int) -> np.ndarray:
        dvxc_R = self.dvxct_sR[spin]
        eig_n = []
        nt_R = ut2_nR.desc.new(dtype=float).empty()
        for ut_R in ut2_nR.data:
            nt_R.data[:] = 0.0
            add_to_density(1.0, ut_R, nt_R.data)
            eig_n.append(dvxc_R.integrate(nt_R))
        return np.array(eig_n)


def number_of_non_empty_bands(ibzwfs: PWFDIBZWaveFunctions,
                              tolerance: float = 1e-5) -> int:
    nocc = 0
    for wfs in ibzwfs:
        nocc = max(nocc, (wfs.occ_n > tolerance).sum())
    return int(ibzwfs.kpt_comm.max_scalar(nocc))


def ibz2bz(ibzwfs: PWFDIBZWaveFunctions,
           setups: Setups,
           relpos_ac: np.ndarray,
           spin: int) -> tuple[int, list[Psi]]:
    nocc = number_of_non_empty_bands(ibzwfs)
    ibz = ibzwfs.ibz
    nbzk = len(ibz.bz)
    comm = ibzwfs.comm
    symmetries = ibzwfs.ibz.symmetries
    rank_K = np.zeros(nbzk, int)
    psit_KnG = {}
    for wfs_s in ibzwfs.wfs_qs:
        wfs = wfs_s[spin].collect(0, nocc)
        if wfs is None:
            continue
        for K, k in enumerate(ibz.bz2ibz_K):
            if k != wfs.k:
                continue
            s = ibz.s_K[K]
            U_cc = symmetries.rotation_scc[s]
            complex_conjugate = ibz.time_reversal_K[K]
            psit1_nG = wfs.psit_nX
            psit2_nG = psit1_nG.transform(U_cc, complex_conjugate)
            psit_KnG[K] = psit2_nG
            rank_K[K] = comm.rank

    comm.sum(rank_K)

    nblocks = max(1, 5 * comm.size // nbzk)
    blocksize = (nocc + nblocks - 1) // nblocks
    nanb_i = [(min(i * blocksize, nocc), min((i + 1) * blocksize, nocc))
              for i in range(nblocks)]

    psit_K = {}
    requests = []
    for K, psit_nG in sorted(psit_KnG.items()):
        for i in range(nblocks):
            rank = (i + nblocks * K) % comm.size
            na, nb = nanb_i[i]
            if rank != comm.rank:
                requests.append(
                    comm.send(psit_nG.data[na:nb], rank, block=False))

    comm.waitall(requests)

    pw = ibzwfs.wfs_qs[0][0].psit_nX.desc.new(comm=None)
    _, occ_skn = ibzwfs.get_all_eigs_and_occs()
    for iK in range(comm.rank, nblocks * nbzk, comm.size):
        K, i = divmod(iK, nblocks)
        na, nb = nanb_i[i]
        print(K, i, na, nb)
        rank = rank_K[K]
        if rank == comm.rank:
            psit_nG = psit_KnG[K][na:nb]
        else:
            psit_nG = pw.new(kpt=ibz.bz.kpt_Kc[K]).empty(nb - na)
            comm.receive(psit_nG.data, rank)
        pt_aiG = psit_nG.desc.atom_centered_functions(
            [setup.pt_j for setup in setups],
            relpos_ac)
        P_ani = pt_aiG.integrate(psit_nG)
        k = ibz.bz2ibz_K[K]
        psit_K[K] = Psi(psit_nG, P_ani, occ_skn[spin, k, na:nb])
    return nocc, psit_K


def nsc_corrections(density: Density,
                    pot_calc: PlaneWavePotentialCalculator
                    ) -> tuple[UGArray, AtomArrays]:
    """Semi-local XC-potential corrections.

    Pseudo-part (calculated from ``density.nt_sR``):::

        ~  _    ~      _    ~     _
       Δv (r) = v     (r) - v    (r),
         σ       σ,HSE       σ,xc

    and PAW corrections:::

         a     / a  a a _   /~a  a ~a _
       Δv    = |φ Δv φ dr - |φ Δv  φ dr,
         σij   / i  σ j     / i  σ  j

    using (calculated from ``density.D_asii``):::

        a _      a     _      a   _
       Δv (r) = v     (r) - v    (r).
         σ       σ,HSE       σ,xc
    """
    xc = pot_calc.xc
    hse = create_functional('HSE06', pot_calc.fine_grid)
    nt_sr = density.nt_sR.interpolate(grid=pot_calc.fine_grid)
    _, dvt_sr, _ = hse.calculate(nt_sr)
    _, vxct_sr, _ = xc.calculate(nt_sr)
    dvt_sr.data -= vxct_sr.data
    dvt_sR = dvt_sr.fft_restrict(grid=density.nt_sR.desc)

    dV_asii = density.D_asii.new()
    for setup, D_sii, dV_sii in zip(pot_calc.setups,
                                    density.D_asii.values(),
                                    dV_asii.values()):
        D_sp = np.array([pack_density(D_ii.real) for D_ii in D_sii])
        dV_sp = np.zeros_like(D_sp)
        xc.calculate_paw_correction(setup, D_sp, dV_sp)
        dV_sp *= -1
        hse.calculate_paw_correction(setup, D_sp, dV_sp)
        dV_sii[:] = unpack_hermitian(dV_sp)

    return dvt_sR, dV_asii
