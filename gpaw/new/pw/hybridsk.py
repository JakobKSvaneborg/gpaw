from __future__ import annotations

from math import nan

import numpy as np
from gpaw.core import PWArray, PWDesc, UGArray, UGDesc
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.pwacf import PWAtomCenteredFunctions
from gpaw.hybrids.paw import pawexxvv
from gpaw.mpi import broadcast
from gpaw.new import zips as zip
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pw.hybrids import truncated_coulomb
from gpaw.new.pw.nschse import Psit, ibz2bz
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.setup import Setups
from gpaw.utilities import unpack_hermitian


class PWHybridHamiltonianK(PWHamiltonian):
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
        assert len(ibzwfs.ibz) % self.kpt_comm.size == 0

        D_aii = D_asii[:, spin].copy()
        if ibzwfs.nspins == 1:
            D_aii = D_aii.copy()
            D_aii.data *= 0.5

        for wfs_s in ibzwfs.wfs_qs:
            wfs = wfs_s[spin]
            if np.allclose(wfs.psit_nX.desc.kpt_c, psit2_nG.desc.kpt_c):
                pt_aiG = wfs.pt_aiX
                assert isinstance(pt_aiG, PWAtomCenteredFunctions)
                weight = wfs.weight
                break
        else:  # no break
            assert False, f'k-point not found: {psit2_nG.desc.kpt_c}'

        evv, evc, ekin = self._apply1(spin, D_aii, pt_aiG,
                                      psit2_nG, Htpsit2_nG,
                                      wfs.myocc_n, calculate_energy, F_av)
        if calculate_energy:
            for name, e in [('hybrid_xc', evv + evc),
                            ('hybrid_kinetic_correction', ekin)]:
                e *= ibzwfs.spin_degeneracy * weight
                self.xc.energies[name] += e
            self.xc.energies['hybrid_xc'] += self.exx_cc

    def _apply1(self,
                spin: int,
                D_aii,
                pt_aiG: PWAtomCenteredFunctions,
                psit_nG: PWArray,
                Htpsit_nG: PWArray | None,
                f_n: np.ndarray,
                calculate_energy: bool,
                F_av=None) -> tuple[float, float, float]:
        comm = self.comm
        band_comm = self.band_comm
        domain_comm = psit_nG.desc.comm

        P_ani = pt_aiG.integrate(psit_nG)

        V0_ani = P_ani.new()

        evv = 0.0
        evc = 0.0
        ekin = 0.0
        for a, D_ii in D_aii.items():
            VV_ii = pawexxvv(self.VV_app[a], D_ii)
            VC_ii = self.VC_aii[a]
            V_ii = -VC_ii - 2 * VV_ii
            V0_ani[a] = P_ani[a] @ V_ii
            if calculate_energy:
                ec = (D_ii * VC_ii).sum()
                ev = (D_ii * VV_ii).sum()
                ekin += ec + 2 * ev
                evv -= ev
                evc -= ec
            elif F_av is not None:
                for psit in self.mypsits:
                    dP_anvi = psit.dP_anvi
                    assert dP_anvi is not None
                    F_av[a] += 4 * np.einsum(
                        'ni, nvi, n -> v',
                        psit.P_ani[a] @ V_ii,
                        dP_anvi[a].conj(),
                        psit.f_n).real

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
                        data = (psit2_nG, P2_ani, f_n, spin)

                rank = (brank + krank * band_comm.size) * domain_comm.size
                psit2_nG, P2_ani, f2_n, s = broadcast(data, rank, comm)
                V_nG = psit2_nG.new()
                V_nG.data[:] = 0.0
                V_ani = P2_ani.new()
                V_ani.data[:] = 0.0
                e += self._apply2(psit2_nG, P2_ani, s, V_nG, V_ani, f2_n,
                                  calculate_energy, F_av)
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

        if not calculate_energy:
            return nan, nan, nan

        evv = domain_comm.sum_scalar(evv)
        evc = domain_comm.sum_scalar(evc)
        ekin = domain_comm.sum_scalar(ekin)

        # e = comm.sum_scalar(e) / domain_comm.size / band_comm.size
        evv += 0.5 * e
        ekin -= e

        return evv, evc, ekin

    def _apply2(self,
                psit2_nG: PWArray,
                P2_ani: AtomArrays,
                spin: int,
                Htpsit2_nG,
                V2_ani,
                f2_n: np.ndarray,
                calculate_energy: bool,
                F_av=None) -> float:
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
                    calculate_energy, F_av)

        e *= -self.exx_fraction / self.nbzk
        return self.comm.sum_scalar(e)

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
                F_av: np.ndarray | None) -> float:
        ut1_nR = psit1.ut_nR
        Q1_aniL = psit1.Q_aniL
        f1_n = psit1.f_n
        ghat_aLG = self.setups.create_compensation_charges(pw, self.relpos_ac)
        v2_G = Htpsit2_nG.desc.empty()
        e = 0.0
        for n1, ut1_R in enumerate(ut1_nR.data):
            rhot_nR = ut2_nR.copy()
            rhot_nR.data *= ut1_R.conj()
            Q_anL = {}
            for a, Q1_niL in Q1_aniL.items():
                Q_anL[a] = P2_ani[a] @ Q1_niL[n1]
            rhot_nG = pw.empty(len(rhot_nR))
            rhot_nR.fft(out=rhot_nG, plan=self.plan)
            ghat_aLG.add_to(rhot_nG, Q_anL)
            if not calculate_energy:
                rhot_nG.data *= v_G
                if F_av is not None:
                    forces(ghat_aLG, rhot_nG, P2_ani,
                           Q_anL,
                           f1_n[n1], f2_n, self.delta_aiiL,
                           psit1.dP_anvi,
                           n1, F_av)
                    continue
            else:
                for rhot_G, f2 in zip(rhot_nG, f2_n):
                    a_G = rhot_G.copy()
                    rhot_G.data *= v_G
                    e12 = a_G.integrate(rhot_G).real * f2 * f1_n[n1]
                    e += e12
            V2_anL = ghat_aLG.integrate(rhot_nG)
            rhot_nG.ifft(out=rhot_nR)
            rhot_nR.data *= ut1_R.data
            x = self.exx_fraction * f1_n[n1] / self.nbzk
            for v2_R, Htpsit2_G in zip(rhot_nR, Htpsit2_nG):
                v2_R.fft(out=v2_G)
                Htpsit2_G.data -= v2_G.data * x
            for a, Q1_niL in Q1_aniL.items():
                V2_ani[a][:] -= x * V2_anL[a] @ Q1_niL[n1].T.conj()
        return e


def forces(ghat_aLG, vrhot2_nG, P2_ani, Q2_anL, f1, f2_n, delta_aiiL,
           dP_anvi, n1, F_av):
    # k-point weight????????
    f12_n = f1 * f2_n
    for a, F_nvL in ghat_aLG.derivative(vrhot2_nG).items():
        F_av[a] -= np.einsum('n, nL, nvL -> v',
                             f12_n,
                             Q2_anL[a].conj(),
                             F_nvL).real
    for a, F_nL in ghat_aLG.integrate(vrhot2_nG).items():
        F_iin = delta_aiiL[a] @ F_nL.T
        F_av[a] -= np.einsum('ijn, vi, nj, n -> v',
                             F_iin,
                             dP_anvi[a][n1].conj(),
                             P2_ani[a],
                             f12_n).real
