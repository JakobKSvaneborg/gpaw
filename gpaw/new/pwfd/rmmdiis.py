from __future__ import annotations

from pprint import pformat
from functools import partial
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
        self._allocate_work_arrays(ibzwfs, shape=(2, self.blocksize))

    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        """Do one step ...

        Also take a look at ths document:

            https://gpaw.readthedocs.io/documentation/rmm-diis.html
        """
        xp = wfs.xp

        psit_nX = wfs.psit_nX
        nbands = psit_nX.dims[0]  # number of bands
        eig_n = xp.empty(nbands)
        mynbands = psit_nX.mydims[0]

        psit2_nX = psit_nX.new(data=self.work_arrays[0])
        psit3_nX = psit_nX.new(data=self.work_arrays[1])

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

        if domain_comm.rank == 0:
            eig_n[:] = xp.asarray(wfs.eig_n)

        Ht = partial(Ht, out=residual_nX, spin=wfs.spin)
        dH = partial(dH, spin=wfs.spin)
        calculate_residuals(residual_nX, dH, dS_aii, wfs, P2_ani, P3_ani)

        error = 0.0
        for n1 in range(0, mynbands, self.blocksize):
            n2 = min(n1 + self.blocksize, mynbands)
            error = self._block_step(weight_n[n1:n2])
        return error

    def block_step(self, weight_n) -> float:
        error = weight_n @ as_np(residual_nX.norm2())
        if wfs.ncomponents == 4:
            error = error.sum()

        self.preconditioner(psit_nX, residual_nX, out=psit2_nX)
        errors_x[:] = 0.0
        for n in range(n1, n2):
            weight = weights[n]
            errors_x[n - n1] = weight * integrate(Rb.array[n - n1],
                                                  Rb.array[n - n1])
        comm.sum(errors_x)
        error += np.sum(errors_x)

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
        for lam, psit_G, dpsit_G, R_G, dR_G in zip(
            lam_x, psitb.array,
            dpsit.array, Rb.array,
            dR.array):
            axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
            axpy(lam, dR_G, R_G)  # R_G += lam * dR_G
        self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)
        lam_x[:] = self.trial_step
        for lam, psit_G, dpsit_G in zip(lam_x, psitb.array, dpsit.array):
            axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
