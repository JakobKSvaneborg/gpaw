"""Non self-consistent HSE06 eigenvalues."""
from __future__ import annotations

from pathlib import Path
from time import time
from typing import IO, Sequence

import numpy as np
from ase.units import Ha

from gpaw.core import PWArray, PWDesc, UGArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.hybrids.paw import pawexxvv
from gpaw.mpi import broadcast
from gpaw.new import zips as zip
from gpaw.new.c import add_to_density
from gpaw.new.calculation import DFTCalculation
from gpaw.new.density import Density
from gpaw.new.logger import Logger
from gpaw.new.pw.hybrids import Psit, ibz2bz, truncated_coulomb
from gpaw.new.pw.pot_calc import PlaneWavePotentialCalculator
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.xc import create_functional
from gpaw.setup import Setups
from gpaw.utilities import pack_density, unpack_hermitian
from gpaw.hybrids import parse_name
from gpaw.new.brillouin import MonkhorstPackKPoints


class NonSelfConsistentHSE06:
    @classmethod
    def from_dft_calculation(cls,
                             dft: DFTCalculation,
                             xc: str = 'HSE06',
                             log: str | Path | IO[str] | None = '-',
                             ) -> NonSelfConsistentHSE06:
        """Create HSE06-eigenvalue calculator from DFT calculation."""
        return cls(dft.ibzwfs,  # type: ignore [arg-type]
                   dft.density,
                   dft.pot_calc,
                   dft.setups,
                   dft.relpos_ac,
                   xc,
                   log)

    def __init__(self,
                 ibzwfs: PWFDIBZWaveFunctions,
                 density: Density,
                 pot_calc: PlaneWavePotentialCalculator,
                 setups: Setups,
                 relpos_ac: np.ndarray,
                 xc: str,
                 log: str | Path | IO[str] | None = '-'):
        assert isinstance(ibzwfs, PWFDIBZWaveFunctions)
        semilocal_xc_name, self.exx_fraction, exx_omega, yukawa = \
            parse_name(xc)
        self.comm = ibzwfs.comm
        self.log = Logger(log, self.comm)
        self.grid = density.nt_sR.desc.new(dtype=complex, comm=None)
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.nbzk = len(ibzwfs.ibz.bz)
        xp = np
        self.plan = self.grid.fft_plans(xp=xp)

        self.mypsits, self.nocc = ibz2bz(
            ibzwfs, setups, relpos_ac, self.grid, self.plan, self.log)

        self.ghat_aLR = setups.create_compensation_charges(
            self.grid, relpos_ac)
        self.relpos_ac = relpos_ac
        self.setups = setups

        self.dxc_sR, self.dhyb_sR, self.dxc_asii, self.dhyb_asii = \
            nsc_corrections(density, pot_calc, semilocal_xc_name)

        for a, D_sii in density.D_asii.items():
            setup = setups[a]
            VC_ii = unpack_hermitian(setup.X_p * self.exx_fraction)
            for D_ii, dhyb_ii in zip(D_sii, self.dhyb_asii[a]):
                VV_ii = self.exx_fraction * (
                    pawexxvv(2 * setup.M_pp, D_ii / ibzwfs.spin_degeneracy))
                dhyb_ii -= VC_ii + VV_ii

        mp = ibzwfs.ibz.bz
        assert isinstance(mp, MonkhorstPackKPoints)
        self.coulomb = truncated_coulomb(
            self.grid.cell_cv, mp.size_c, exx_omega, yukawa)

    def calculate(self,
                  ibzwfs: PWFDIBZWaveFunctions,
                  na: int = 0,
                  nb: int = 0,
                  ibz_indices: Sequence[int] | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues at several k-points.

        Returns DFT and HSE06 eigenvalues in eV.
        """
        eig_sin, dxc_sin, dhyb_sin = self._calculate(
            ibzwfs, na, nb, ibz_indices)
        return eig_sin, eig_sin - dxc_sin + dhyb_sin

    def _calculate(self,
                   ibzwfs: PWFDIBZWaveFunctions,
                   na: int = 0,
                   nb: int = 0,
                   ibz_indices: Sequence[int] | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nb = nb if nb > 0 else ibzwfs.nbands + nb

        comm = self.comm
        domain_comm = ibzwfs.domain_comm
        band_comm = ibzwfs.band_comm
        kpt_comm = ibzwfs.kpt_comm

        if ibz_indices is None:
            ibz_indices = range(len(ibzwfs.ibz))
        nkpts = len(ibz_indices)

        comm_rank_is = np.zeros((nkpts, ibzwfs.nspins), int)
        for i, k in enumerate(ibz_indices):
            for spin in range(ibzwfs.nspins):
                if ibzwfs.rank_ks[k, spin] == kpt_comm.rank:
                    if band_comm.rank == 0 and domain_comm.rank == 0:
                        comm_rank_is[i, spin] = comm.rank
        comm.sum(comm_rank_is)

        self.log('Calculating eigenvalues:')
        self.log('  k-points:', nkpts)
        self.log(f'  Bands: {na}-{nb - 1} (inclusive)')

        tb = 0.0
        t1 = time()
        eig_isn = []  # self-consistent DFT eigenvalues
        dxc_isn = []  # DFT eigenvalue changes
        dhyb_isn = []  # Hybrid eigenvalue changes
        for i, k in enumerate(ibz_indices):
            eig_sn = []
            dxc_sn = []
            dhyb_sn = []
            for spin in range(ibzwfs.nspins):
                data = None
                tb -= time()
                if ibzwfs.rank_ks[k, spin] == kpt_comm.rank:
                    wfs = ibzwfs._get_wfs(k, spin).collect(na, nb)
                    if wfs is not None:
                        data = (wfs.psit_nX,
                                wfs.P_ani,
                                wfs.eig_n * Ha,
                                spin)
                psit_nG, P_ani, eig_n, spin = broadcast(
                    data, comm_rank_is[i, spin], comm=comm)
                tb += time()
                eig_sn.append(eig_n)
                dxc_n, dhyb_n = self.calculate_one_kpt(psit_nG, P_ani, spin)
                dxc_sn.append(dxc_n)
                dhyb_sn.append(dhyb_n)
            eig_isn.append(eig_sn)
            dxc_isn.append(dxc_sn)
            dhyb_isn.append(dhyb_sn)
        t2 = time()
        self.log(f'  Seconds: {t2 - t1:.3f} '
                 f'(wave-function broadcasting: {tb:.3f} seconds)')

        eig_sin = np.array(eig_isn).transpose((1, 0, 2))
        dxc_sin = np.array(dxc_isn).transpose((1, 0, 2))
        dhyb_sin = np.array(dhyb_isn).transpose((1, 0, 2))
        deig_sin = dhyb_sin - dxc_sin

        self.log('HSE06-eigenvalue shifts:')
        self.log(f'  min: {deig_sin.min():.3f} eV')
        self.log(f'  ave: {deig_sin.mean():.3f} eV')
        self.log(f'  max: {deig_sin.max():.3f} eV')

        return eig_sin, dxc_sin, dhyb_sin

    def calculate_one_kpt(self,
                          psit2_nG: PWArray,
                          P2_ani: AtomArrays,
                          spin: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalue-contributions at one k-point.

        Returned eigenvalue-contributions are in eV.
        """
        ut2_nR = self.grid.empty(len(psit2_nG))
        psit2_nG.ifft(out=ut2_nR, plan=self.plan, periodic=False)

        dxc_n, dhyb_n = self._semi_local_xc_parts(ut2_nR, spin)

        # PAW corrections:
        for a, dxc_sii in self.dxc_asii.items():
            P2_ni = P2_ani[a]
            dxc_n += np.einsum('ni, ij, nj -> n',
                               P2_ni.conj(), dxc_sii[spin], P2_ni).real
            dhyb_sii = self.dhyb_asii[a]
            dhyb_n += np.einsum('ni, ij, nj -> n',
                                P2_ni.conj(), dhyb_sii[spin], P2_ni).real

        domain_comm = self.dxc_asii.layout.atomdist.comm
        domain_comm.sum(dxc_n)
        domain_comm.sum(dhyb_n)

        pw2 = psit2_nG.desc
        eig_n = np.zeros(len(psit2_nG))
        for psit1 in self.mypsits:
            if psit1.spin == spin:
                pw = pw2.new(kpt=pw2.kpt_c - psit1.kpt_c)
                v_G = self.coulomb(pw)
                eig_n += self._exx_part(pw, v_G, psit1, ut2_nR, P2_ani)
        eig_n *= -self.exx_fraction / self.nbzk
        self.comm.sum(eig_n)
        dhyb_n += eig_n

        return dxc_n * Ha, dhyb_n * Ha

    def _exx_part(self,
                  pw: PWDesc,
                  v_G: np.ndarray,
                  psit1: Psit,
                  ut2_nR: UGArray,
                  P2_ani: AtomArrays) -> np.ndarray:
        """EXX contribution from one k-point in the BZ."""
        ut1_nR = psit1.ut_nR
        Q1_aniL = psit1.Q_aniL
        f1_n = psit1.f_n
        phase_a = np.exp(-2j * np.pi * (self.relpos_ac @ pw.kpt_c))
        ghat_aLG = self.setups.create_compensation_charges(
            pw, self.relpos_ac)
        e_n = np.zeros(len(ut2_nR))
        for n1, ut1_R in enumerate(ut1_nR.data):
            rhot_nR = ut2_nR.copy()
            rhot_nR.data *= ut1_R.conj()
            Q_anL = {}
            if 1:
                for a, Q1_niL in Q1_aniL.items():
                    Q_anL[a] = P2_ani[a] @ Q1_niL[n1]
                rhot_nG = pw.empty(len(rhot_nR))
                rhot_nR.fft(out=rhot_nG, plan=self.plan)
                ghat_aLG.add_to(rhot_nG, Q_anL)
            else:
                for a, Q1_niL in Q1_aniL.items():
                    Q_anL[a] = P2_ani[a] @ Q1_niL[n1] * phase_a[a]
                self.ghat_aLR.add_to(rhot_nR, Q_anL)
                rhot_nG = pw.empty(len(rhot_nR))
                rhot_nR.fft(out=rhot_nG, plan=self.plan)
            rhot_nG.data *= v_G**0.5
            e_n += rhot_nG.norm2() * f1_n[n1]
        return e_n

    def _semi_local_xc_parts(self,
                             ut2_nR: UGArray,
                             spin: int) -> tuple[np.ndarray, np.ndarray]:
        dxc_n = np.zeros(len(ut2_nR))
        dhyb_n = np.zeros(len(ut2_nR))
        if self.dxc_sR is not None:
            dxc_R = self.dxc_sR[spin]
            dhyb_R = self.dhyb_sR[spin]
            nt_R = ut2_nR.desc.new(dtype=float).empty()
            for n, ut_R in enumerate(ut2_nR.data):
                nt_R.data[:] = 0.0
                add_to_density(1.0, ut_R, nt_R.data)
                dxc_n[n] = dxc_R.integrate(nt_R)
                dhyb_n[n] = dhyb_R.integrate(nt_R)
        return dxc_n, dhyb_n


def nsc_corrections(density: Density,
                    pot_calc: PlaneWavePotentialCalculator,
                    semilocal_xc_name: str
                    ) -> tuple[UGArray, UGArray, AtomArrays, AtomArrays]:
    """Semi-local XC-potential corrections.

    Pseudo-part (calculated from ``density.nt_sR``):::

       ~     _
       v    (r)
        σ,XC

    and PAW corrections:::

        a     / a a a _   /~a a ~a _
       v    = |φ v φ dr - |φ v  φ dr,
        σij   / i σ j     / i σ  j

    using (calculated from ``density.D_asii``):::

        a    _
       v    (r)
        σ,XC
    """
    nt_sr = density.nt_sR.interpolate(grid=pot_calc.fine_grid)
    xc = pot_calc.xc
    _, dxc_sr, _ = xc.calculate(nt_sr)
    hyb = create_functional(semilocal_xc_name, pot_calc.fine_grid)
    _, dhyb_sr, _ = hyb.calculate(nt_sr)
    dxc_sr = dxc_sr.gather()
    dhyb_sr = dhyb_sr.gather()
    if dxc_sr is not None:
        grid = density.nt_sR.desc.new(comm=None)
        dxc_sR = dxc_sr.fft_restrict(grid=grid)
        dhyb_sR = dhyb_sr.fft_restrict(grid=grid)
    else:
        dxc_sR = None
        dhyb_sR = None
    dxc_asii = density.D_asii.new()
    dhyb_asii = density.D_asii.new()
    for a, D_sii in density.D_asii.items():
        setup = pot_calc.setups[a]
        D_sp = np.array([pack_density(D_ii.real) for D_ii in D_sii])
        dV_sp = np.zeros_like(D_sp)
        xc.calculate_paw_correction(setup, D_sp, dV_sp)
        dxc_asii[a][:] = unpack_hermitian(dV_sp)
        dV_sp[:] = 0.0
        hyb.calculate_paw_correction(setup, D_sp, dV_sp)
        dhyb_asii[a][:] = unpack_hermitian(dV_sp)
        if setup.hubbard_u is not None:
            _, dHU_sii = setup.hubbard_u.calculate(setup, D_sii)
            dxc_asii[a][:] += dHU_sii

    return dxc_sR, dhyb_sR, dxc_asii, dhyb_asii
