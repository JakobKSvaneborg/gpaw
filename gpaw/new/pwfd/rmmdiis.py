from __future__ import annotations

from functools import partial
from pprint import pformat

import numpy as np

from gpaw.core import PWDesc
from gpaw.gpu import as_np
from gpaw.new import zips as zip
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals


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
        self._allocate_work_arrays(ibzwfs, shape=(2,))

    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        """Do one step ...

        See here:

            https://gpaw.readthedocs.io/documentation/rmm-diis.html
        """

        psit_nX = wfs.psit_nX
        mynbands = psit_nX.mydims[0]

        work_nX = psit_nX.new(data=self.work_arrays[0, :mynbands])
        residual_nX = psit_nX.new(data=self.work_arrays[1, :mynbands])

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=work_nX.data,
                                 Htpsit_nX=residual_nX)
        dH = partial(dH, spin=wfs.spin)
        calculate_residuals(residual_nX, dH, dS_aii, wfs, P2_ani, P3_ani)

        # domain_comm = psit_nX.desc.comm
        # band_comm = psit_nX.comm
        # is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        # if domain_comm.rank == 0:
        #    eig_n[:] = xp.asarray(wfs.eig_n)

        work2_nX = work_nX.desk.empty(self.blocksize)

        error = 0.0
        for n1 in range(0, mynbands, self.blocksize):
            n2 = min(n1 + self.blocksize, mynbands)
            error += block_step(
                wfs,
                weight_n[n1:n2],
                psit_nX[n1:n2],
                residual_nX[n1:n2],
                work_nX[:n2 - n1],
                work2_nX[:n2 - n1],
                Ht,
                dH,
                dS_aii,
                P2_ani[:, n2 - n1],
                P3_ani[:, n2 - n1],
                self.preconditioner)
        return error


def block_step(wfs,
               weight_n,
               psit_nX,
               residual_nX,
               work1_nX,
               work2_nX,
               Ht,
               dH,
               dS_aii,
               P2_ani,
               P3_ani,
               preconditioner) -> float:
    error = weight_n @ as_np(residual_nX.norm2())

    presidual_nX = work1_nX
    dresidual_nX = work2_nX
    preconditioner(psit_nX, residual_nX, out=presidual_nX)

    Ht(presidual_nX, out=dresidual_nX, spin=wfs.spin)
    wfs.pt_aiX.integrate(presidual_nX, out=P2_ani)
    dH(P2_ani, out_ani=P3_ani)
    calculate_residuals(dresidual_nX, dH, dS_aii, wfs, P2_ani, P3_ani)

    # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
    a_n = [d_X.integrate(r_X) for d_X, r_X in zip(dresidual_nX, residual_nX)]
    b_n = dresidual_nX.norm2()
    lambda_n = -a_n / b_n
    presidual_nX.data.T *= lambda_n
    psit_nX.data += presidual_nX.data
    dresidual_nX.data.T *= lambda_n
    residual_nX.data += dresidual_nX.data
    preconditioner(psit_nX, residual_nX, out=presidual_nX)
    presidual_nX.data.T *= lambda_n
    psit_nX.data += presidual_nX.data

    return error
