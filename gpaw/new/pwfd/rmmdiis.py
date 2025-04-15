from __future__ import annotations

import warnings
from pprint import pformat

import numpy as np
from gpaw.core import PWDesc
from gpaw.gpu import as_np
from gpaw.new import zips as zip
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions


class RMMDIIS(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 converge_bands='occupied',
                 blocksize=None,
                 niter: int = 1,
                 scalapack_parameters=None):
        if niter != 1:
            warnings.warn(f'Ignoring niter={niter} in RMMDIIS')
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
        self._allocate_work_arrays(ibzwfs, shape=(2,))

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        """Do one step ...

        See here:

            https://gpaw.readthedocs.io/documentation/rmm-diis.html
        """

        psit_nX = wfs.psit_nX
        mynbands = psit_nX.mydims[0]

        work_nX = psit_nX.new(data=self.work_arrays[0, :mynbands])
        residual_nX = psit_nX.new(data=self.work_arrays[1, :mynbands])

        P_ani = wfs.P_ani
        work1_ani = P_ani.new()
        work2_ani = P_ani.new()

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=work_nX.data,
                                 Htpsit_nX=residual_nX)
        calculate_residuals(wfs.psit_nX, residual_nX, wfs.pt_aiX,
                            wfs.P_ani, wfs.myeig_n,
                            dH, dS_aii, work1_ani, work2_ani)

        # domain_comm = psit_nX.desc.comm
        # band_comm = psit_nX.comm
        # is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        # if domain_comm.rank == 0:
        #    eig_n[:] = xp.asarray(wfs.eig_n)

        n = min(self.blocksize, mynbands)
        work2_nX = work_nX.desc.empty(n)
        P1_ani = P_ani.layout.empty(n)
        P2_ani = P_ani.layout.empty(n)

        if weight_n is None:
            error = np.inf
        else:
            error = weight_n @ as_np(residual_nX.norm2())
        for n1 in range(0, mynbands, self.blocksize):
            n2 = n1 + self.blocksize
            if n2 > mynbands:
                n2 = mynbands
                P1_ani = P1_ani[:, :n2 - n1]
                P2_ani = P2_ani[:, :n2 - n1]
            block_step(
                psit_nX[n1:n2],
                residual_nX[n1:n2],
                wfs.pt_aiX, wfs.myeig_n[n1:n2], Ht, dH, dS_aii,
                work_nX[:n2 - n1],
                work2_nX[:n2 - n1],
                P1_ani, P2_ani,
                self.preconditioner)
        wfs._P_ani = None
        wfs.orthonormalized = False
        wfs.orthonormalize(work_nX.data)
        return error


def block_step(psit_nX,
               R_nX,
               pt_aiX,
               eig_n,
               Ht,
               dH,
               dS_aii,
               work1_nX,
               work2_nX,
               P1_ani,
               P2_ani,
               preconditioner) -> None:
    PR_nX = work1_nX
    dR_nX = work2_nX
    ekin_n = preconditioner(psit_nX, R_nX, out=PR_nX)

    Ht(PR_nX, out=dR_nX)
    P_ani = pt_aiX.integrate(PR_nX)
    calculate_residuals(PR_nX, dR_nX, pt_aiX, P_ani, eig_n,
                        dH, dS_aii, P1_ani, P2_ani)
    a_n = [-d_X.integrate(r_X)
           for d_X, r_X in zip(dR_nX, R_nX)]
    b_n = dR_nX.norm2()
    shape = (len(a_n),) + (1,) * (psit_nX.data.ndim - 1)
    lambda_n = (a_n / b_n).reshape(shape)
    PR_nX.data *= lambda_n
    psit_nX.data += PR_nX.data
    dR_nX.data *= lambda_n
    R_nX.data += dR_nX.data
    preconditioner(psit_nX, R_nX, out=PR_nX, ekin_n=ekin_n)
    PR_nX.data *= 0.1
    psit_nX.data += PR_nX.data
