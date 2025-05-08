from __future__ import annotations

import warnings
from pprint import pformat

import numpy as np
from gpaw.gpu import as_np
from gpaw.new import zips as zip
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.pwfd.davidson import sliced_preconditioner

class RMMDIIS(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 hamiltonian,
                 converge_bands='occupied',
                 niter: int = 1,
                 trial_step: float | None = None,
                 scalapack_parameters=None,
                 max_buffer_mem: int = 200 * 1024 ** 2):
        """RMM-DIIS eigensolver.

        Solution steps are:

        * Subspace diagonalization
        * Calculation of residuals
        * Improvement of wave functions:  psi' = psi + lambda PR + lambda PR'
        * Orthonormalization

        Parameters
        ==========
        trial_step:
            Step length for final step.  Use None for using the previously
            optimized step lengths.
        """

        if niter != 1:
            warnings.warn(f'Ignoring niter={niter} in RMMDIIS')
        super().__init__(hamiltonian, converge_bands,
                         max_buffer_mem=max_buffer_mem)
        self.trial_step = trial_step

    def __str__(self):
        return pformat(dict(name='RMMDIIS',
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        self._allocate_work_arrays(ibzwfs, shape=(1,))
        self._allocate_buffer_arrays(ibzwfs, shape=(2,))

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        """Do one step ...

        See here:

            https://gpaw.readthedocs.io/documentation/rmm-diis.html
        """

        psit_nX = wfs.psit_nX
        mynbands = psit_nX.mydims[0]

        residual_nX = psit_nX.new(data=self.work_arrays[0, :mynbands])

        P_ani = wfs.P_ani
        work1_ani = P_ani.new()
        work2_ani = P_ani.new()

        wfs.subspace_diagonalize(Ht, dH,
                                 psit2_nX=residual_nX,
                                 data_buffer=self.data_buffers[0])
        calculate_residuals(wfs.psit_nX, residual_nX, wfs.pt_aiX,
                            wfs.P_ani, wfs.myeig_n,
                            dH, dS_aii, work1_ani, work2_ani)

        work1_nX = psit_nX.create_work_buffer(self.data_buffers[0])
        ekin_n = psit_nX.norm2('kinetic')
        sliced_preconditioner(psit_nX, residual_nX, work1_nX, self.preconditioner)
        P_ani = wfs.pt_aiX.integrate(residual_nX)
        
        work2_nX = psit_nX.create_work_buffer(self.data_buffers[1])
        blocksize = work1_nX.data.shape[0]
        P1_ani = P_ani.layout.empty(blocksize)
        P2_ani = P_ani.layout.empty(blocksize)
        if weight_n is None:
            error = np.inf
        else:
            error = weight_n @ as_np(residual_nX.norm2())

        comm = psit_nX.comm
        blocksize_world = comm.sum_scalar(blocksize)
        totalbands = comm.sum_scalar(mynbands)
        for i1, N1 in enumerate(
                range(0, totalbands, blocksize_world)):
            n1 = i1 * blocksize
            n2 = n1 + blocksize
            #sP_ani = {a: P_ni[n1:n2] for a, P_ni in P_ani.items()}
            sP_ani = P_ani[:, n1:n2]
            if n2 > mynbands:
                n2 = mynbands
                P1_ani = P1_ani[:, :n2 - n1]
                P2_ani = P2_ani[:, :n2 - n1]
            block_step(
                psit_nX[n1:n2],
                residual_nX[n1:n2],
                sP_ani, wfs.myeig_n[n1:n2], Ht, dH, dS_aii,
                self.trial_step,
                work1_nX[:n2 - n1],
                work2_nX[:n2 - n1],
                P1_ani, P2_ani,
                ekin_n[n1:n2],
                wfs.pt_aiX,
                self.preconditioner)
        wfs._P_ani = None
        wfs.orthonormalized = False
        wfs.orthonormalize(residual_nX)
        return error


def block_step(psit_nX,
               PR_nX,
               P_ani,
               eig_n,
               Ht,
               dH,
               dS_aii,
               trial_step,
               work1_nX,
               work2_nX,
               P1_ani,
               P2_ani,
               ekin_n,
               pt_aiX,
               preconditioner) -> None:
    """See here:

            https://gpaw.readthedocs.io/documentation/rmm-diis.html
    """
    buff_nX = work1_nX
    dR_nX = work2_nX

    Ht(PR_nX, out=dR_nX)
    calculate_residuals(PR_nX, dR_nX, pt_aiX, P_ani, eig_n,
                        dH, dS_aii, P1_ani, P2_ani)
    preconditioner(psit_nX, PR_nX, out=buff_nX,
                   ekin_n=ekin_n, inverse=True)
    a_n = psit_nX.xp.asarray(
        [-d_X.integrate(r_X) for d_X, r_X in zip(dR_nX, buff_nX)])
    b_n = dR_nX.norm2()

    preconditioner(psit_nX, dR_nX, out=buff_nX, ekin_n=ekin_n)
    shape = (len(a_n),) + (1,) * (psit_nX.data.ndim - 1)
    lambda_n = (a_n / b_n).reshape(shape)
    PR_nX.data *= lambda_n
    psit_nX.data += PR_nX.data
    buff_nX.data *= lambda_n
    #R_nX.data += dR_nX.data
    PR_nX.data += buff_nX.data
    if trial_step is None:
        PR_nX.data *= lambda_n
    else:
        PR_nX.data *= trial_step
    psit_nX.data += PR_nX.data
