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
                 converge_bands='occupied',
                 blocksize=None,
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
        work1_ani = P_ani.new()
        work2_ani = P_ani.new()

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=work_nX.data,
                                 Htpsit_nX=residual_nX)
        dH = partial(dH, spin=wfs.spin)
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

        error = 0.0
        for n1 in range(0, mynbands, self.blocksize):
            n2 = min(n1 + self.blocksize, mynbands)
            error += block_step(
                wfs,
                weight_n[n1:n2],
                psit_nX[n1:n2],
                residual_nX[n1:n2],
                Ht, dH, dS_aii,
                work_nX[:n2 - n1],
                work2_nX[:n2 - n1],
                P1_ani, P2_ani,
                self.preconditioner)
        return error


def block_step(wfs,
               weight_n,
               psit_nX,
               residual_nX,
               Ht,
               dH,
               dS_aii,
               work1_nX,
               work2_nX,
               P1_ani,
               P2_ani,
               preconditioner) -> float:
    error = weight_n @ as_np(residual_nX.norm2())

    presidual_nX = work1_nX
    dresidual_nX = work2_nX
    preconditioner(psit_nX, residual_nX, out=presidual_nX)

    Ht(presidual_nX, out=dresidual_nX, spin=wfs.spin)
    wfs.pt_aiX.integrate(presidual_nX, out=P1_ani)
    calculate_residuals(dresidual_nX, dH, dS_aii, wfs, P1_ani, P2_ani)
    a_n = [-d_X.integrate(r_X) for d_X, r_X in zip(dresidual_nX, residual_nX)]
    b_n = dresidual_nX.norm2()
    lambda_n = (a_n / b_n).reshape((-1,) + (1,) * (psit_nX.data.ndim - 1))
    print(a_n, b_n, lambda_n)
    1 / 0
    presidual_nX.data *= lambda_n
    psit_nX.data += presidual_nX.data
    dresidual_nX.data *= lambda_n
    residual_nX.data += dresidual_nX.data
    preconditioner(psit_nX, residual_nX, out=presidual_nX)
    presidual_nX.data *= lambda_n
    psit_nX.data += presidual_nX.data

    return error
