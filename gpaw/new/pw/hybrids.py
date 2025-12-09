from __future__ import annotations

from dataclasses import dataclass
from math import pi
from time import time

import numpy as np
from gpaw.core import PWArray, PWDesc, UGArray, UGDesc
from gpaw.core.arrays import XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.pwacf import PWAtomCenteredFunctions
from gpaw.hybrids.paw import pawexxvv
from gpaw.mpi import broadcast
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.logger import Logger
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new import zips as zip
from gpaw.setup import Setups
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.blas import mmm
from scipy.linalg.blas import get_blas_funcs


@dataclass
class Psit:
    ut_nR: UGArray
    P_ani: AtomArrays
    f_n: np.ndarray
    kpt_c: np.ndarray
    Q_aniL: dict[int, np.ndarray]
    spin: int
    dP_anvi: AtomArrays | None = None  # used for forces


def truncated_coulomb(pw: PWDesc,
                      omega: float = 0.11,
                      yukawa: bool = False) -> np.ndarray:
    """Fourier transform of truncated Coulomb.

    Real space:::

        erfc(ωr)
        --------.
           r

    Reciprocal space:::

        4π             _ _ 2     2
      ------(1 - exp(-(G+k) /(4 ω )))
       _ _ 2
      (G+k)

    (G+k=0 limit is pi/ω^2).
    """
    G2_G = pw.ekin_G * 2
    if yukawa:
        v_G = 4 * pi / (G2_G + omega**2)
    else:
        v_G = 4 * pi * (1 - np.exp(-G2_G / (4 * omega**2)))
        ok_G = G2_G > 1e-10
        v_G[ok_G] /= G2_G[ok_G]
        v_G[~ok_G] = pi / omega**2
    return v_G


def number_of_non_empty_bands(ibzwfs: PWFDIBZWaveFunctions,
                              tolerance: float = 1e-5) -> int:
    nocc = 0
    for wfs in ibzwfs:
        nocc = max(nocc, int((wfs.occ_n > tolerance).sum()))
    return int(ibzwfs.kpt_comm.max_scalar(nocc))


def ibz2bz(ibzwfs: PWFDIBZWaveFunctions,
           setups: Setups,
           relpos_ac: np.ndarray,
           grid: UGDesc,
           plan,  # FFT-plan
           log: Logger | None = None,
           forces: bool = False) -> tuple[list[Psit], int]:
    """Compute BZ from IBZ and distribute."""
    log = log or Logger(None, None)
    nocc = number_of_non_empty_bands(ibzwfs)
    nspins = ibzwfs.nspins
    ibz = ibzwfs.ibz
    log(ibz)
    log('Occupied bands:', nocc)

    log('Transforming wave functions from IBZ to BZ: ', end='')
    t1 = time()
    nbzk = len(ibz.bz)
    comm = ibzwfs.comm
    symmetries = ibzwfs.ibz.symmetries
    rank_Ks = np.zeros((nbzk, nspins), int)
    kpt_Kc = np.zeros((nbzk, 3))
    psit_KsnG = {}
    for wfs1 in ibzwfs:
        wfs = wfs1.collect(0, nocc)
        if wfs is None:
            continue
        for K, k in enumerate(ibz.bz2ibz_K):
            if k != wfs.k:
                continue
            rank_Ks[K, wfs.spin] = comm.rank
            s = ibz.s_K[K]
            U_cc = symmetries.rotation_scc[s]
            complex_conjugate = ibz.time_reversal_K[K]
            psit1_nG = wfs.psit_nX
            assert isinstance(psit1_nG, PWArray)
            psit2_nG = psit1_nG.transform(U_cc, complex_conjugate)
            if wfs.spin == 0:
                kpt_Kc[K] = psit2_nG.desc.kpt_c
            assert abs(psit2_nG.desc.kpt_c - ibz.bz.kpt_Kc[K]).max() < 1e-8
            psit_KsnG[(K, wfs.spin)] = psit2_nG
    comm.sum(rank_Ks)
    comm.sum(kpt_Kc)
    t2 = time()
    log(f'{t2 - t1:.3f} seconds')

    nocc_total = nocc * nbzk
    blocksize = (nocc_total + comm.size - 1) // comm.size
    blocks = []
    for rank in range(comm.size):
        Ka, na = divmod(rank * blocksize, nocc)
        Kb, nb = divmod((rank + 1) * blocksize, nocc)
        for K in range(Ka, min(Kb, nbzk)):
            blocks.append((rank, K, (na, nocc)))
            na = 0
        if nb > na and Kb < nbzk:
            blocks.append((rank, Kb, (na, nb)))

    log('Distributing wave functions and iFFT-ing to real space: ', end='')
    t1 = time()
    requests = []
    for (K, spin), psit_nG in psit_KsnG.items():
        for rank, KK, (na, nb) in blocks:
            if KK != K:
                continue
            if rank != comm.rank:
                requests.append(
                    comm.send(psit_nG.data[na:nb], rank,
                              block=False, tag=K * nspins + spin))

    pw = ibzwfs._wfs_u[0].psit_nX.desc.new(comm=None)
    _, occ_skn = ibzwfs.get_all_eigs_and_occs(broadcast=True)

    mypsits = []
    for rank, K, (na, nb) in blocks:
        if rank != comm.rank:
            continue
        pt_aiG = None
        for spin in range(nspins):
            if rank_Ks[K, spin] == rank:
                psit_nG = psit_KsnG[(K, spin)][na:nb]
            else:
                psit_nG = pw.new(kpt=kpt_Kc[K]).empty(nb - na)
                comm.receive(psit_nG.data, rank_Ks[K, spin],
                             tag=K * nspins + spin)
            pt_aiG = pt_aiG or psit_nG.desc.atom_centered_functions(
                [setup.pt_j for setup in setups],
                relpos_ac)
            P_ani = pt_aiG.integrate(psit_nG)

            psit_nR = psit_nG.ifft(grid=grid, plan=plan, periodic=False)
            Q_aniL = {a: np.einsum('ijL, nj -> niL',
                                   setup.Delta_iiL, P_ani[a].conj())
                      for a, setup in enumerate(setups)}
            k = ibz.bz2ibz_K[K]
            f_n = occ_skn[spin, k, na:nb]
            psit = Psit(psit_nR, P_ani, f_n, psit_nG.desc.kpt_c, Q_aniL, spin)
            if forces:
                psit.dP_anvi = pt_aiG.derivative(psit_nG)
            mypsits.append(psit)

    comm.waitall(requests)

    t2 = time()
    log(f'{t2 - t1:.3f} seconds')

    return mypsits, nocc


class PWHybridHamiltonian(PWHamiltonian):
    band_local = False

    def __init__(self,
                 grid: UGDesc,
                 pw: PWDesc,
                 xc,
                 setups: Setups,
                 relpos_ac,
                 atomdist,
                 log,
                 kpt_comm,
                 band_comm,
                 comm):
        super().__init__(grid, pw.dtype)
        self.pw = pw
        self.exx_fraction = xc.exx_fraction
        self.exx_omega = xc.exx_omega
        self.xc = xc
        self.kpt_comm = kpt_comm
        self.band_comm = band_comm
        self.comm = comm
        self.log = log
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.relpos_ac = relpos_ac
        self.setups = setups

        # Stuff for PAW core-core, core-valence and valence-valence correctios:
        self.exx_cc = sum(setup.ExxC for setup in setups) * self.exx_fraction
        self.VC_aii = [unpack_hermitian(setup.X_p * self.exx_fraction)
                       for setup in setups]
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.VV_app = [setup.M_pp * self.exx_fraction for setup in setups]

        self.mypsits: list[Psit] = []
        self.nbzk = 0
        self.real = np.issubdtype(pw.dtype, np.floating)
        self.zaxpy = get_blas_funcs('axpy', dtype=complex)

    def update_wave_functions(self,
                              ibzwfs: PWFDIBZWaveFunctions,
                              forces=False):
        """Compute BZ from IBZ and distribute over the entire world!"""
        self.mypsits, _ = ibz2bz(
            ibzwfs, self.setups, self.relpos_ac, self.grid_local, self.plan,
            self.log if self.nbzk == 0 else None, forces)
        self.nbzk = len(ibzwfs.ibz.bz)
        self.xc.energies = {'hybrid_xc': 0.0,
                            'hybrid_kinetic_correction': 0.0}

    def move(self, relpos_av: np.ndarray) -> None:
        self.relpos_ac = relpos_av

    def apply_orbital_dependent(self,
                                ibzwfs: IBZWaveFunctions,
                                D_asii,
                                psit2_nG: XArray,
                                spin: int,
                                Htpsit2_nG: XArray | None = None,
                                calculate_energy: bool = False,
                                F_av: np.ndarray | None = None) -> None:
        assert isinstance(psit2_nG, PWArray)
        assert Htpsit2_nG is None or isinstance(Htpsit2_nG, PWArray)
        assert isinstance(ibzwfs, PWFDIBZWaveFunctions)
        assert len(ibzwfs.ibz) * ibzwfs.nspins % self.kpt_comm.size == 0

        domain_comm = psit2_nG.desc.comm

        if F_av is not None:
            F1_av = np.zeros_like(F_av)
        else:
            F1_av = None

        # Find projectors and k-point weight for psit2_nG:
        for wfs in ibzwfs:
            if wfs.spin != spin:
                continue
            if np.allclose(wfs.psit_nX.desc.kpt_c, psit2_nG.desc.kpt_c):
                pt_aiG = wfs.pt_aiX
                assert isinstance(pt_aiG, PWAtomCenteredFunctions)
                kweight = wfs.weight
                break
        else:  # no break
            assert False, f'k-point not found: {psit2_nG.desc.kpt_c}'

        D_aii = D_asii[:, spin].copy()
        if ibzwfs.nspins == 1:
            D_aii = D_aii.copy()
            D_aii.data *= 0.5

        evv = 0.0  # valence-valence contribution
        evc = 0.0  # valence-core contribution
        V_aii = D_aii.new()
        for a, D_ii in D_aii.items():
            VV_ii = pawexxvv(self.VV_app[a], D_ii)
            VC_ii = self.VC_aii[a]
            V_ii = -VC_ii - 2 * VV_ii
            V_aii[a] = V_ii
            if calculate_energy:
                ec = (D_ii * VC_ii).sum()
                ev = (D_ii * VV_ii).sum()
                evv -= ev
                evc -= ec

        # distribute V_aii
        V2_aii = V_aii.gather(broadcast=True)

        if calculate_energy:
            evv = domain_comm.sum_scalar(evv) * self.kpt_comm.size * kweight
            evc = domain_comm.sum_scalar(evc) * self.kpt_comm.size * kweight
        elif F1_av is not None:
            for a, V_ii in V2_aii.items():
                for psit in self.mypsits:
                    dP_anvi = psit.dP_anvi
                    assert dP_anvi is not None
                    force_v = np.einsum('ni, nvi, n -> v',
                                        psit.P_ani[a] @ V_ii,
                                        dP_anvi[a].conj(),
                                        psit.f_n).real
                    force_v = 2 / self.nbzk * force_v
                    F1_av[a] += force_v

        ekin = -evc - 2 * evv

        e = self._apply1(spin, D_aii, pt_aiG,
                         psit2_nG, Htpsit2_nG,
                         kweight, wfs.myocc_n, V_aii,
                         calculate_energy, F1_av)

        evv += 0.5 * e
        ekin -= e

        if calculate_energy:
            for name, e in [('hybrid_xc', evv + evc),
                            ('hybrid_kinetic_correction', ekin)]:
                e *= ibzwfs.spin_degeneracy
                self.xc.energies[name] += e
            self.xc.energies['hybrid_xc'] += self.exx_cc

        if F1_av is not None:
            assert F_av is not None
            F_av += ibzwfs.spin_degeneracy * kweight * F1_av

    def _apply1(self,
                spin: int,
                D_aii,
                pt_aiG: PWAtomCenteredFunctions,
                psit_nG: PWArray,
                Htpsit_nG: PWArray | None,
                kweight: float,
                f_n: np.ndarray,
                V_aii,
                calculate_energy: bool,
                F1_av=None) -> float:
        comm = self.comm
        band_comm = self.band_comm
        domain_comm = psit_nG.desc.comm

        P_ani = pt_aiG.integrate(psit_nG)

        V0_ani = P_ani.new()

        for a, D_ii in D_aii.items():
            V0_ani[a] = P_ani[a] @ V_aii[a]

        e = 0.0
        for krank in range(self.kpt_comm.size):
            for brank in range(band_comm.size):
                data = None
                if krank == self.kpt_comm.rank and brank == band_comm.rank:
                    psit2_nG = psit_nG.gather()
                    P2_ani = P_ani.gather()
                    if psit2_nG is not None:
                        # Remove band_comm so that data can be pickled
                        # when calling broadcast(data, ...) later:
                        psit2_nG = psit2_nG[:]
                        P2_ani = AtomArrays(P2_ani.layout,
                                            dims=(len(P2_ani.data),),
                                            data=P2_ani.data)
                        data = (psit2_nG, P2_ani, f_n, spin, kweight)

                rank = (brank + krank * band_comm.size) * domain_comm.size
                psit2_nG, P2_ani, f2_n, s, w = broadcast(data, rank, comm=comm)
                V_nG = psit2_nG.new()
                V_nG.data[:] = 0.0
                V_ani = P2_ani.new()
                V_ani.data[:] = 0.0
                e += self._apply2(psit2_nG, P2_ani, s, V_nG, V_ani, f2_n,
                                  calculate_energy, F1_av) * w
                if Htpsit_nG is None:
                    continue
                comm.sum(V_nG.data, root=rank)
                comm.sum(V_ani.data, root=rank)
                if krank == self.kpt_comm.rank:
                    if brank == band_comm.rank:
                        V2_nG = Htpsit_nG.new()
                        V2_nG.scatter_from(V_nG)
                        V2_ani = V0_ani.new()
                        V2_ani.scatter_from(V_ani)
                        V2_ani.data += V0_ani.data
                        Htpsit_nG.data += V2_nG.data
                        pt_aiG.add_to(Htpsit_nG, V2_ani)
        return e

    def _apply2(self,
                psit2_nG: PWArray,
                P2_ani: AtomArrays,
                spin: int,
                Htpsit2_nG,
                V2_ani,
                f2_n: np.ndarray,
                calculate_energy: bool,
                F1_av=None) -> float:
        ut2_nR = self.grid_local.empty(len(psit2_nG))
        psit2_nG.ifft(out=ut2_nR, plan=self.plan, periodic=False)

        e = 0.0
        pw2 = psit2_nG.desc
        for psit1 in self.mypsits:
            if psit1.spin == spin:
                pw = pw2.new(kpt=pw2.kpt_c - psit1.kpt_c)
                v_G = truncated_coulomb(pw, self.exx_omega)
                e += self._apply3(
                    pw, v_G, psit1, ut2_nR, P2_ani, Htpsit2_nG, V2_ani, f2_n,
                    calculate_energy, F1_av)

        e *= -self.exx_fraction / self.nbzk
        return self.comm.sum_scalar(e)

    # from line_profiler import profile
    # @profile
    def _apply3(self,
                pw: PWDesc,
                v_G: np.ndarray,
                psit1: Psit,
                ut2_nR: UGArray,
                P2_ani: AtomArrays,
                Htpsit2_nG: PWArray,
                V2_ani,
                f2_n: np.ndarray,
                calculate_energy: bool,
                F1_av: np.ndarray | None) -> float:
        ut1_nR = psit1.ut_nR
        Q1_aniL = psit1.Q_aniL
        f1_n = psit1.f_n
        ghat_aLG = self.setups.create_compensation_charges(pw, self.relpos_ac)
        ghat_aLG._lazy_init()
        ghat_GA = ghat_aLG._lfc.expand(cc=not self.real)
        N2 = len(ut2_nR)
        Q_anL = ghat_aLG.layout.empty(N2)
        rhot2_nG = pw.empty(N2)
        tmp_Q = self.plan.tmp_Q
        tmp_R = self.plan.tmp_R
        eikR_a = ghat_aLG._lfc.eikR_a
        pw2 = Htpsit2_nG.desc
        NR = tmp_R.size
        NG = pw.myshape[0]
        NG2 = pw2.myshape[0]
        tmp_G = np.empty(NG, complex)
        Q_G = pw.indices(tmp_Q.shape)
        Q2_G = pw2.indices(tmp_Q.shape)
        e = 0.0
        for n1, ut1_R in enumerate(ut1_nR.data):
            f1 = f1_n[n1]
            for a, Q1_niL in Q1_aniL.items():
                Q_anL[a] = P2_ani[a] @ Q1_niL[n1] * eikR_a[a].conj()
            if self.real:
                mmm(1.0 / pw.dv, Q_anL.data, 'N', ghat_GA, 'T',
                    0.0, rhot2_nG.data.view(float))
            else:
                mmm(1.0 / pw.dv, Q_anL.data, 'N', ghat_GA, 'C',
                    0.0, rhot2_nG.data)
            for n2, (rhot_G, ut2_R) in enumerate(zip(rhot2_nG.data,
                                                     ut2_nR.data)):
                tmp_R[:] = ut2_R
                tmp_R *= ut1_R.conj()
                self.plan.fft()
                a_G = tmp_Q.ravel()[Q_G]
                self.zaxpy(a_G, rhot_G, NG, 1.0 / NR)
                if not calculate_energy:
                    rhot_G *= v_G
                else:
                    tmp_G[:] = rhot_G
                    rhot_G *= v_G
                    e12 = tmp_G.view(float) @ rhot_G.view(float)
                    if self.real:
                        e12 = 2 * e12 - (tmp_G[0] * rhot_G[0]).real
                    e += e12 * f2_n[n2] * f1 * pw.dv
            if F1_av is not None:
                forces(ghat_aLG, rhot2_nG, P2_ani,
                       Q_anL,
                       f1, f2_n, self.nbzk, self.delta_aiiL,
                       psit1.dP_anvi,
                       n1, eikR_a, F1_av)
                continue
            if self.real:
                ghat_GA[0] *= 0.5
                mmm(2.0, rhot2_nG.data.view(float), 'N', ghat_GA, 'N',
                    0.0, Q_anL.data)
                ghat_GA[0] *= 2.0
            else:
                mmm(1.0, rhot2_nG.data, 'N', ghat_GA, 'N', 0.0, Q_anL.data)
            x = self.exx_fraction * f1 / self.nbzk
            for rhot_G, Htpsit2_G in zip(rhot2_nG.data, Htpsit2_nG.data):
                self.plan.ifft_sphere(rhot_G, pw)
                tmp_R *= ut1_R.data
                self.plan.fft()
                # Htpsit2_G -= x / NR * pw2.cut(tmp_Q)
                v2_G = tmp_Q.ravel()[Q2_G]
                self.zaxpy(v2_G, Htpsit2_G, NG2, -x / NR)
            for a, Q1_niL in Q1_aniL.items():
                V2_ani[a] -= x * Q_anL[a] @ Q1_niL[n1].T.conj() * eikR_a[a]
        return e


def forces(ghat_aLG, vrhot2_nG, P2_ani, Q2_anL, f1, f2_n, nbzk, delta_aiiL,
           dP_anvi, n1, eikR_a, F_av):
    f12_n = f1 * f2_n
    for a, F_nvL in ghat_aLG.derivative(vrhot2_nG).items():
        F_av[a] -= 0.25 / nbzk * np.einsum('n, nL, nvL -> v',
                                           f12_n,
                                           (Q2_anL[a] * eikR_a[a]).conj(),
                                           F_nvL).real
    for a, F_nL in ghat_aLG.integrate(vrhot2_nG).items():
        F_iin = delta_aiiL[a] @ F_nL.T
        F_av[a] -= 0.5 / nbzk * np.einsum('ijn, vi, nj, n -> v',
                                          F_iin,
                                          dP_anvi[a][n1],
                                          P2_ani[a].conj(),
                                          f12_n).real
