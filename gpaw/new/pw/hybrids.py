from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from gpaw.core import PWArray, PWDesc, UGDesc
from gpaw.core.arrays import DistributedArrays as XArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.hybrids.paw import pawexxvv
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.new import zips
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.typing import Array1D
from gpaw.utilities import unpack_hermitian
from gpaw.utilities.blas import mmm


def coulomb(pw: PWDesc,
            grid: UGDesc,
            omega: float) -> PWArray:
    if omega == 0.0:
        wstc = WignerSeitzTruncatedCoulomb(
            pw.cell_cv, np.array([1, 1, 1]))
        return wstc.get_potential_new(pw, grid)

    v_G = pw.empty()
    G2_G = pw.ekin_G * 2
    v_G.data[:] = 4 * pi * (1 - np.exp(-G2_G / (4 * omega**2)))
    if pw.ng1 == 0:
        v_G.data[1:] /= G2_G[1:]
        v_G.data[0] = pi / omega**2
    else:
        v_G.data /= G2_G

    return v_G


@dataclass
class Psi:
    psit_nG: PWArray
    P_ani: AtomArrays
    f_n: Array1D | None = None

    def empty(self, nbands):
        return Psi(self.psit_nG.desc.empty(dims=nbands),
                   self.P_ani.layout.empty(dims=nbands),
                   np.empty(nbands))


class PWHybridHamiltonian(PWHamiltonian):
    def __init__(self,
                 grid: UGDesc,
                 pw: PWDesc,
                 xc,
                 setups,
                 fracpos_ac,
                 atomdist):
        super().__init__(grid, pw)
        self.grid = grid
        self.pw = pw
        self.exx_fraction = xc.exx_fraction
        self.exx_omega = xc.exx_omega

        # Stuff for PAW core-core, core-valence and valence-valence correctios:
        self.exx_cc = sum(setup.ExxC for setup in setups) * self.exx_fraction
        self.VC_aii = [unpack_hermitian(setup.X_p * self.exx_fraction)
                       for setup in setups]
        self.delta_aiiL = [setup.Delta_iiL for setup in setups]
        self.VV_app = [setup.M_pp * self.exx_fraction for setup in setups]

        self.v_G = coulomb(pw, grid, self.exx_omega)
        self.v_G.data *= self.exx_fraction

        self.ghat_aLG = setups.create_compensation_charges(
            pw, fracpos_ac, atomdist)
        self.ghat_aLG._lazy_init()
        self.ghat_GA = self.ghat_aLG._lfc.expand()

        self.plan = grid.fft_plans()

    def apply_orbital_dependent(self,
                                ibzwfs: IBZWaveFunctions,
                                D_asii,
                                psit2_nG: XArray,
                                spin: int,
                                Htpsit2_nG: XArray) -> None:
        assert isinstance(psit2_nG, PWArray)
        assert isinstance(Htpsit2_nG, PWArray)
        wfs = ibzwfs.wfs_qs[0][spin]
        D_aii = D_asii[:, spin]
        if ibzwfs.nspins == 1:
            D_aii.data *= 0.5
        psi1 = Psi(wfs.psit_nX, wfs.P_ani, wfs.myocc_n)
        pt_aiG = wfs.pt_aiX

        if psi1.psit_nG is psit2_nG:
            # We are doing a subspace diagonalization ...
            evv, evc, ekin = self.apply_same(D_aii, pt_aiG,
                                             psi1, Htpsit2_nG)
            for name, e in [('exx_vv', evv),
                            ('exx_vc', evc),
                            ('exx_kinetic', ekin)]:
                e *= ibzwfs.spin_degeneracy
                if spin == 0:
                    ibzwfs.energies[name] = e
                else:
                    ibzwfs.energies[name] += e
            ibzwfs.energies['exx_cc'] = self.exx_cc
            return

        # We are applying the exchange operator (defined by psit1_nG,
        # P1_ani, f1_n and D_aii) to another set of wave functions
        # (psit2_nG):
        psi2 = Psi(psit2_nG, pt_aiG.integrate(psit2_nG))
        self.apply_other(D_aii, pt_aiG,
                         psi1, psi2,
                         Htpsit2_nG)

    def apply_same(self,
                   D_aii,
                   pt_aiG,
                   psi: Psi,
                   Htpsit_nG: PWArray) -> tuple[float, float, float]:
        comm = Htpsit_nG.comm
        mynbands = len(Htpsit_nG)

        evv = 0.0
        evc = 0.0
        ekin = 0.0
        B_ani = {}
        for a, D_ii in D_aii.items():
            VV_ii = pawexxvv(self.VV_app[a], D_ii)
            VC_ii = self.VC_aii[a]
            B_ni = psi.P_ani[a] @ (-VC_ii - 2 * VV_ii)
            B_ani[a] = B_ni
            ec = (D_ii * VC_ii).sum()
            ev = (D_ii * VV_ii).sum()
            ekin += ec + 2 * ev
            evv -= ev
            evc -= ec

        Q_nA = np.empty((mynbands, 9 * D_aii.natoms))
        Q_aniL = {a: np.einsum('ijL, nj -> niL',
                               delta_iiL, psi.P_ani[a])
                  for a, delta_iiL in enumerate(self.delta_aiiL)}

        tmp1_nR = self.grid.empty(mynbands)
        rhot_nR = self.grid.empty(mynbands)
        rhot_nG = self.pw.empty(mynbands)
        vrhot_G = self.pw.empty()

        psi2 = psi.empty(nbands=self.pw.maxmysize)
        for p in range(comm.size // 2 + 1):
            if p < comm.size // 2:
                psi.send((comm.rank + p + 1) % comm.size)
                psi2.receive((comm.rank - p - 1) % comm.size)
            psit1_nR = ifft(psi.psit_nG, tmp1_nR, self.plan)
            if p == 0:
                psit2_nR = psit1_nR
                P2_ani = psi.P_ani
                f2_n = psi.f_n
            for n2, (psit2_R, out_G) in enumerate(zips(psit2_nR, Htpsit_nG)):
                rhot_nR.data[:] = psit1_nR.data * psit2_R.data
                fft(rhot_nR, rhot_nG, plan=self.plan)
                A1 = 0
                for a, Q_niL in Q_aniL.items():
                    A2 = A1 + 9
                    Q_nA[:, A1:A2] = P2_ani[a][n2] @ Q_niL
                # Note that G runs over G0.real, G0.imag, G1.real, G1.imag, ...
                mmm(1.0, Q_nA, 'N', self.ghat_GA, 'T',
                    1.0, rhot_nG.data.view(float))
                for n1, (rhot_R, rhot_G, f1) in enumerate(zips(rhot_nR,
                                                               rhot_nG,
                                                               psi.f_n)):
                    vrhot_G.data = rhot_G.data * self.v_G.data
                    A_aL = self.ghat_aLG.integrate(vrhot_G)
                    for a, A_L in A_aL.items():
                        B_ani[a][n2] -= np.einsum(
                            'L, ijL, j -> i',
                            f1 * A_L, self.delta_aiiL[a], psi.P_ani[a][n1])
                    e = f1 * f2_n[n2] * rhot_G.integrate(vrhot_G)
                    evv -= 0.5 * e
                    ekin += e
                    rhot_G.data[:] = vrhot_G.data
                ifft(rhot_nG, rhot_nR, plan=self.plan)
                rhot_nR.data *= psit1_nR.data
                fft(rhot_nR, rhot_nG, self.plan)
                out_G.data -= rhot_nG.data.sum(0) * f1

        pt_aiG.add_to(Htpsit_nG, B_ani)

        return evv, evc, ekin

    def apply_other(self,
                    D_aii,
                    pt_aiG,
                    psit1_nG: PWArray,
                    P1_ani: AtomArrays,
                    f1_n: Array1D,
                    psit2_nG: PWArray,
                    P2_ani: AtomArrays,
                    Htpsit2_nG: PWArray) -> None:
        n1 = len(psit1_nG)
        n2 = len(psit1_nG)

        # psit1_R = self.tmp1_nR[:n1]...grid.empty()
        psit1_R = self.grid.empty()
        psit2_R = self.grid.empty()
        rhot_R = self.grid.empty()
        rhot_G = self.pw.empty()

        B_ani = {}
        for a, D_ii in D_aii.items():
            VV_ii = pawexxvv(self.VV_app[a], D_ii)
            VC_ii = self.VC_aii[a]
            B_ni = P2_ani[a] @ (-VC_ii - 2 * VV_ii)
            B_ani[a] = B_ni

        for n2, (psit2_G, out_G) in enumerate(zips(psit2_nG, Htpsit2_nG)):
            psit2_G.ifft(out=psit2_R, plan=self.plan)

            for n1, (psit1_G, f1) in enumerate(zips(psit1_nG, f1_n)):
                psit1_G.ifft(out=psit1_R, plan=self.plan)
                rhot_R.data[:] = psit1_R.data * psit2_R.data
                rhot_R.fft(out=rhot_G, plan=self.plan)
                Q_aL = {a: np.einsum('i, ijL, j -> L',
                                     P1_ani[a][n1], delta_iiL, P2_ani[a][n2])
                        for a, delta_iiL in enumerate(self.delta_aiiL)}
                self.ghat_aLG.add_to(rhot_G, Q_aL)
                rhot_G.data *= self.v_G.data
                A_aL = self.ghat_aLG.integrate(rhot_G)
                rhot_G.ifft(out=rhot_R, plan=self.plan)
                rhot_R.data *= psit1_R.data
                rhot_R.fft(out=rhot_G, plan=self.plan)
                out_G.data -= rhot_G.data * f1

                for a, A_L in A_aL.items():
                    B_ani[a][n2] -= np.einsum(
                        'L, ijL, j -> i',
                        f1 * A_L, self.delta_aiiL[a], P1_ani[a][n1])

        pt_aiG.add_to(Htpsit2_nG, B_ani)


def ifft(psit_nG, out_nR, plan):
    for psit_G, out_R in zips(psit_nG, out_nR):
        psit_G.ifft(out=out_R, plan=plan)
    return out_nR


def fft(rhot_nR, rhot_nG, plan):
    for rhot_R, rhot_G in zips(rhot_nR, rhot_nG):
        rhot_R.fft(out=rhot_G, plan=plan)
