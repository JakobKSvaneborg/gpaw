from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import nan, pi
from pathlib import Path
from time import time
from typing import IO

import numpy as np
from ase.units import Ha
from gpaw.core import PWArray, PWDesc, UGArray, UGDesc
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.hybrids.paw import pawexxvv
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.mpi import broadcast
from gpaw.new import zips as zip
from gpaw.new.c import add_to_density
from gpaw.new.calculation import DFTCalculation
from gpaw.new.density import Density
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.logger import Logger
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pw.hybrids import fft, pawexxvv, truncated_coulomb
from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.xc import create_functional
from gpaw.setup import Setups
from gpaw.typing import Array1D
from gpaw.utilities import pack_density, unpack_hermitian
from gpaw.utilities.blas import mmm


@dataclass
class Psit:
    ut_nR: UGArray
    P_ani: AtomArrays
    f_n: np.ndarray
    kpt_c: np.ndarray
    Q_aniL: dict[int, np.ndarray]
    spin: int


class PWHybridHamiltonianK(PWHamiltonian):
    def __init__(self,
                 grid: UGDesc,
                 pw: PWDesc,
                 xc,
                 setups,
                 relpos_ac,
                 atomdist,
                 log,
                 comm):
        super().__init__(grid, pw)
        self.pw = pw
        self.exx_fraction = xc.exx_fraction
        self.exx_omega = xc.exx_omega
        self.xc = xc
        self.comm = comm
        self.log = log
        self.cgrid = grid.new(dtype=complex, comm=None)
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        xp = np
        self.plan = self.cgrid.fft_plans(xp=xp)

        self.relpos_ac = relpos_ac
        self.setups = setups

        # Stuff for PAW core-core, core-valence and valence-valence correctios:
        self.exx_cc = sum(setup.ExxC for setup in setups) * self.exx_fraction
        self.VC_aii = [unpack_hermitian(setup.X_p * self.exx_fraction)
                       for setup in setups]
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.VV_app = [setup.M_pp * self.exx_fraction for setup in setups]

        self.ghat_aLG = setups.create_compensation_charges(
            pw, relpos_ac, atomdist)

        self.mypsits = []
        self.nocc = -1

    def apply_orbital_dependent(self,
                                ibzwfs: IBZWaveFunctions,
                                D_asii,
                                psit2_nG: XArray,
                                spin: int,
                                Htpsit2_nG: XArray) -> None:
        assert isinstance(psit2_nG, PWArray)
        assert isinstance(Htpsit2_nG, PWArray)
        D_aii = D_asii[:, spin].copy()
        if ibzwfs.nspins == 1:
            D_aii = D_aii.copy()
            D_aii.data *= 0.5

        for wfs_s in ibzwfs.wfs_qs:
            wfs = wfs_s[spin]
            if np.allclose(wfs.psit_nX.desc.kpt_c, psit2_nG.desc.kpt_c):
                pt_aiG = wfs.pt_aiX
                break
        else:  # no break
            1 / 0

        if len(self.mypsits) == 0:
            self.mypsits, self.nocc = ibz2bz(
                ibzwfs, self.setups, self.relpos_ac, self.cgrid, self.plan,
                self.log)

            assert wfs.psit_nX.data is psit2_nG.data

            # We are doing a subspace diagonalization ...
            evv, evc, ekin = self.apply(spin, D_aii, pt_aiG,
                                        psit2_nG, Htpsit2_nG)
            for name, e in [('hybrid_xc', evv + evc),
                            ('hybrid_kinetic_correction', ekin)]:
                e *= ibzwfs.spin_degeneracy
                if spin == 0:
                    self.xc.energies[name] = e
                else:
                    self.xc.energies[name] += e
            self.xc.energies['hybrid_xc'] += self.exx_cc
            return

        # We are applying the exchange operator (defined by psit1_nG,
        # P1_ani, f1_n and D_aii) to another set of wave functions
        # (psit2_nG):
        self.apply1(D_aii, pt_aiG, psit2_nG, Htpsit2_nG)

    def apply(self,
              D_aii,
              pt_aiG,
              psit_nG: PWArray,
              Htpsit_nG: PWArray) -> tuple[float, float, float]:
        comm = self.comm
        band_comm = ibzwfs.band_comm
        kpt_comm = ibzwfs.kpt_comm

        kpt_comm_rank_k = ibzwfs.rank_k
        comm_rank_k = np.zeros(len(kpt_comm_rank_k), int)
        for k, kpt_comm_rank in enumerate(kpt_comm_rank_k):
            if kpt_comm_rank == kpt_comm.rank:
                if band_comm.rank == 0 and domain_comm.rank == 0:
                    comm_rank_k[k] = comm.rank
        comm.sum(comm_rank_k)

        tb = 0.0
        t1 = time()
        eig_ksn = []  # self-consistent DFT eigenvalues
        deig_ksn = []  # HSE06 eigenvalue changes
        for rank in range(kpt_comm.size):
            data = None
            tb -= time()
            if rank == kpt_comm.rank:
                psit_nG = psit_nG.gather()
                if psit_nG is not None:
                    data = (psit_nG, spin)
            psit_nG, s = broadcast(data, rank * band_comm.size, comm)
            tb += time()
            self.apply1(psit_nG, P_ani, s)
        t2 = time()
        self.log(f'  Seconds: {t2 - t1:.3f} '
                 f'(wave-function broadcasting: {tb:.3f} seconds)')

    def calculate_one_kpt(self,
                          psit2_nG: PWArray,
                          P2_ani: AtomArrays,
                          spin: int) -> np.ndarray:
        """Calculate eigenvalues at one k-point.

        Returned eigenvalues are in eV.
        """
        ut2_nR = self.grid.empty(len(psit2_nG))
        psit2_nG.ifft(out=ut2_nR, plan=self.plan, periodic=False)

        deig_n = self._semi_local_xc_part(ut2_nR, spin)

        # PAW corrections:
        for a, dE_sii in self.dE_asii.items():
            P2_ni = P2_ani[a]
            deig_n += np.einsum('ni, ij, nj -> n',
                                P2_ni.conj(), dE_sii[spin], P2_ni).real

        self.dE_asii.layout.atomdist.comm.sum(deig_n)

        pw2 = psit2_nG.desc
        eig_n = np.zeros(len(psit2_nG))
        for psit1 in self.mypsits:
            if psit1.spin == spin:
                pw = pw2.new(kpt=pw2.kpt_c - psit1.kpt_c)
                v_G = truncated_coulomb(pw, self.hse06_omega)
                eig_n += self._exx_part(v_G, psit1, ut2_nR, P2_ani)
        eig_n *= -self.exx_fraction / self.nbzk
        self.comm.sum(eig_n)

        return (deig_n + eig_n) * Ha

    def _exx_part(self,
                  v_G: PWArray,
                  psit1: Psit,
                  ut2_nR: UGArray,
                  P2_ani: AtomArrays) -> np.ndarray:
        """EXX contribution from one k-point in the BZ."""
        ut1_nR = psit1.ut_nR
        Q1_aniL = psit1.Q_aniL
        f1_n = psit1.f_n
        phase_a = np.exp(-2j * np.pi * (self.relpos_ac @ v_G.desc.kpt_c))
        ghat_aLG = self.setups.create_compensation_charges(
            v_G.desc, self.relpos_ac)
        e_n = np.zeros(len(ut2_nR))
        for n1, ut1_R in enumerate(ut1_nR.data):
            rhot_nR = ut2_nR.copy()
            rhot_nR.data *= ut1_R.conj()
            Q_anL = {}
            if 1:
                for a, Q1_niL in Q1_aniL.items():
                    Q_anL[a] = P2_ani[a] @ Q1_niL[n1]
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
            e_n += rhot_nG.norm2() * f1_n[n1]
        return e_n

    def _semi_local_xc_part(self,
                            ut2_nR: UGArray,
                            spin: int) -> np.ndarray:
        eig_n = np.zeros(len(ut2_nR))
        if self.dvxct_sR is not None:
            dvxc_R = self.dvxct_sR[spin]
            nt_R = ut2_nR.desc.new(dtype=float).empty()
            for n, ut_R in enumerate(ut2_nR.data):
                nt_R.data[:] = 0.0
                add_to_density(1.0, ut_R, nt_R.data)
                eig_n[n] = dvxc_R.integrate(nt_R)
        return eig_n


def ibz2bz(ibzwfs: PWFDIBZWaveFunctions,
           setups: Setups,
           relpos_ac: np.ndarray,
           grid: UGDesc,
           plan,  # FFT-plan
           log: Logger) -> tuple[list[Psit], int]:
    """Compute BZ from IBZ and distribute."""
    nocc = number_of_non_empty_bands(ibzwfs)
    ibz = ibzwfs.ibz
    log(ibz)
    log('Occupied bands:', nocc)

    log('Transforming wave functions from IBZ to BZ: ', end='')
    t1 = time()
    nbzk = len(ibz.bz)
    comm = ibzwfs.comm
    symmetries = ibzwfs.ibz.symmetries
    rank_K = np.zeros(nbzk, int)
    kpt_Kc = np.zeros((nbzk, 3))
    psit_KsnG = {}
    for wfs_s in ibzwfs.wfs_qs:
        wfs_s = [wfs.collect(0, nocc) for wfs in wfs_s]
        if wfs_s[0] is None:
            continue
        for K, k in enumerate(ibz.bz2ibz_K):
            if k != wfs_s[0].k:
                continue
            rank_K[K] = comm.rank
            s = ibz.s_K[K]
            U_cc = symmetries.rotation_scc[s]
            complex_conjugate = ibz.time_reversal_K[K]
            psit_snG = []
            for wfs in wfs_s:
                psit1_nG = wfs.psit_nX
                psit2_nG = psit1_nG.transform(U_cc, complex_conjugate)
                psit_snG.append(psit2_nG)
                kpt_Kc[K] = psit2_nG.desc.kpt_c
            psit_KsnG[K] = psit_snG
    comm.sum(rank_K)
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
    for K, psit_snG in psit_KsnG.items():
        for rank, KK, (na, nb) in blocks:
            if KK != K:
                continue
            if rank != comm.rank:
                for psit_nG in psit_snG:
                    requests.append(
                        comm.send(psit_nG.data[na:nb], rank,
                                  block=False, tag=K))

    pw = ibzwfs.wfs_qs[0][0].psit_nX.desc.new(comm=None)
    _, occ_skn = ibzwfs.get_all_eigs_and_occs(broadcast=True)

    mypsits = []
    for rank, K, (na, nb) in blocks:
        if rank != comm.rank:
            continue
        pt_aiG = None
        for spin in range(ibzwfs.nspins):
            if rank_K[K] == rank:
                psit_nG = psit_KsnG[K][spin][na:nb]
            else:
                psit_nG = pw.new(kpt=kpt_Kc[K]).empty(nb - na)
                comm.receive(psit_nG.data, rank_K[K], tag=K)
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
            mypsits.append(psit)

    comm.waitall(requests)

    t2 = time()
    log(f'{t2 - t1:.3f} seconds')

    return mypsits, nocc


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

        a  _     a     _     a   _
       Δv (r) = v     (r) - v    (r).
         σ       σ,HSE       σ,xc
    """
    xc = pot_calc.xc
    hse = create_functional('HSE06', pot_calc.fine_grid)
    nt_sr = density.nt_sR.interpolate(grid=pot_calc.fine_grid)
    _, dvt_sr, _ = hse.calculate(nt_sr)
    _, vxct_sr, _ = xc.calculate(nt_sr)
    dvt_sr.data -= vxct_sr.data
    dvt_sr = dvt_sr.gather()
    if dvt_sr is not None:
        dvt_sR = dvt_sr.fft_restrict(grid=density.nt_sR.desc.new(comm=None))
    else:
        dvt_sR = None
    dV_asii = density.D_asii.new()
    for a, D_sii in density.D_asii.items():
        setup = pot_calc.setups[a]
        D_sp = np.array([pack_density(D_ii.real) for D_ii in D_sii])
        dV_sp = np.zeros_like(D_sp)
        xc.calculate_paw_correction(setup, D_sp, dV_sp)
        dV_sp *= -1
        hse.calculate_paw_correction(setup, D_sp, dV_sp)
        dV_asii[a][:] = unpack_hermitian(dV_sp)

    return dvt_sR, dV_asii
