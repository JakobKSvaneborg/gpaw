from __future__ import annotations

from functools import partial
from pprint import pformat

import numpy as np

from gpaw.new import zips as zip
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.core import PWDesc


class RMMDIIS(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 blocksize=None,
                 converge_bands='occupied',
                 scalapack_parameters=None):
        if blocksize is None:
            if isinstance(wf_grid, PWDesc):
                S = wf_grid.comm.size
                # Use a multiple of S for maximum efficiency
                blocksize = int(np.ceil(10 / S)) * S
            else:
                blocksize = 10
        super().__init__(preconditioner_factory, converge_bands, blocksize)

    def __str__(self):
        return pformat(dict(name='RMMDIIS',
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)

    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        xp = M_nn.xp

        psit_nX = wfs.psit_nX
        B = psit_nX.dims[0]  # number of bands
        eig_N = xp.empty(2 * B)
        b = psit_nX.mydims[0]

        psit2_nX = psit_nX.new(data=self.work_arrays[0, :b])
        psit3_nX = psit_nX.new(data=self.work_arrays[1, :b])

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=psit2_nX.data,
                                 Htpsit_nX=psit3_nX)
        residual_nX = psit3_nX  # will become (H-e*S)|psit> later

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0



        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=psit2_nX.data,
                                 Htpsit_nX=psit3_nX)

        calculate_residuals(residual_nX, dH, dS_aii, wfs, P2_ani, P3_ani)

        comm = wfs.gd.comm

        B = self.blocksize
        dR = R.new(dist=None, nbands=B)
        dpsit = dR.new()
        P = P.new(bcomm=None, nbands=B)
        P2 = P.new()
        errors_x = np.zeros(B)

        Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

        error = 0.0
        for n1 in range(0, wfs.bd.mynbands, B):
            n2 = n1 + B
            if n2 > wfs.bd.mynbands:
                n2 = wfs.bd.mynbands
                B = n2 - n1
                P = P.new(nbands=B)
                P2 = P.new()
                dR = dR.new(nbands=B, dist=None)
                dpsit = dR.new()

            n_x = np.arange(n1, n2)
            psitb = psit.view(n1, n2)

            Rb = R.view(n1, n2)

            errors_x[:] = 0.0
            for n in range(n1, n2):
                weight = weights[n]
                errors_x[n - n1] = weight * integrate(Rb.array[n - n1],
                                                      Rb.array[n - n1])
            comm.sum(errors_x)
            error += np.sum(errors_x)

            # Precondition the residual:
            ekin_x = self.preconditioner.calculate_kinetic_energy(
                psitb.array, kpt)
            self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)

            # Calculate the residual of dpsit_G, dR_G = (H - e S) dpsit_G:
            # self.timer.start('Apply Hamiltonian')
            dpsit.apply(Ht, out=dR)
            # self.timer.stop('Apply Hamiltonian')
            dpsit.matrix_elements(wfs.pt, out=P)

            self.calculate_residuals(kpt, wfs, ham, dpsit,
                                     P, kpt.eps_n[n_x], dR, P2, n_x,
                                     calculate_change=True)

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            RdR_x = np.array([integrate(dR_G, R_G)
                              for R_G, dR_G in zip(Rb.array, dR.array)])
            dRdR_x = np.array([integrate(dR_G, dR_G) for dR_G in dR.array])
            comm.sum(RdR_x)
            comm.sum(dRdR_x)
            lam_x = -RdR_x / dRdR_x

            # New trial wavefunction and residual
            for lam, psit_G, dpsit_G, R_G, dR_G in zip(lam_x, psitb.array,
                                                       dpsit.array, Rb.array,
                                                       dR.array):
                axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
                axpy(lam, dR_G, R_G)  # R_G += lam** dR_G

            # Final trial step
            self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)

            for lam, psit_G, dpsit_G in zip(lam_x, psitb.array, dpsit.array):
                axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
