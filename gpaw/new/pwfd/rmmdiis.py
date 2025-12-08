from __future__ import annotations

from pprint import pformat

import numpy as np

from gpaw.gpu import as_np
from gpaw.new import zips as zip
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.utilities import as_dtype_precision


class RMMDIIS(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 hamiltonian,
                 converge_bands='occupied',
                 niter: int = 1,
                 diis_steps: int = 1,
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

        super().__init__(hamiltonian, converge_bands,
                         max_buffer_mem=max_buffer_mem)
        self.trial_step = trial_step
        self.niter = niter
        self.diis_steps = diis_steps

    def __str__(self):
        return pformat(dict(name='RMMDIIS',
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        self._allocate_work_arrays(ibzwfs, shape=(1,))
        self._allocate_buffer_arrays(ibzwfs, shape=(2, self.diis_steps))

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        """
        See here:

            https://gpaw.readthedocs.io/documentation/rmm-diis.html
        """

        psit_nX = wfs.psit_nX
        mynbands = psit_nX.mydims[0]

        residual_nX = psit_nX.new(data=self.work_arrays[0, :mynbands])

        P_ani = wfs.P_ani
        work1_ani = P_ani.new()
        work2_ani = P_ani.new()
        work_nX = psit_nX.create_work_buffer(self.data_buffers[0, 0])
        blocksize = work_nX.data.shape[0]
        P0_ani = P_ani.layout.empty(blocksize)
        P1_ani = P_ani.layout.empty(blocksize)
        P2_ani = P_ani.layout.empty(blocksize)

        for iteration in range(self.niter):
            wfs.subspace_diagonalize(Ht, dH,
                                     psit2_nX=residual_nX,
                                     data_buffer=self.data_buffers[0, 0])
            calculate_residuals(wfs.psit_nX, residual_nX, wfs.pt_aiX,
                                wfs.P_ani, wfs.myeig_n,
                                dH, dS_aii, work1_ani, work2_ani)

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
                n2 = min(n1 + blocksize, mynbands)
                sP_ani = P0_ani[:, :n2 - n1]
                sP1_ani = P1_ani[:, :n2 - n1]
                sP2_ani = P2_ani[:, :n2 - n1]
                self.block_step(
                    psit_nX[n1:n2],
                    residual_nX[n1:n2],
                    sP_ani, wfs.myeig_n[n1:n2], Ht, dH, dS_aii,
                    self.trial_step,
                    self.data_buffers,
                    sP1_ani, sP2_ani,
                    wfs.pt_aiX,
                    self.preconditioner)
            wfs._P_ani = None
            wfs.orthonormalized = False
        wfs.orthonormalize(residual_nX)
        return error

    def block_step(self,
                   psit_nX,
                   R_nX,
                   P_ani,
                   eig_n,
                   Ht,
                   dH,
                   dS_aii,
                   trial_step,
                   data_buffers,
                   P1_ani,
                   P2_ani,
                   pt_aiX,
                   preconditioner) -> None:
        """See here:

                https://gpaw.readthedocs.io/documentation/rmm-diis.html
        """
        xp = psit_nX.xp
        dtype = psit_nX.desc.dtype

        # RMM Part:
        PR_nX = psit_nX.create_work_buffer(self.data_buffers[0, 0])
        dR_nX = psit_nX.create_work_buffer(self.data_buffers[1, 0])

        ekin_n = preconditioner(psit_nX, R_nX, out=PR_nX)
        Ht(PR_nX, out=dR_nX)
        pt_aiX.integrate(PR_nX, out=P_ani)  # XXX: This is expensive
        calculate_residuals(PR_nX, dR_nX, pt_aiX, P_ani, eig_n,
                            dH, dS_aii, P1_ani, P2_ani)
        a_n = psit_nX.xp.asarray(
            [-d_X.integrate(r_X).real for d_X, r_X in zip(dR_nX, R_nX)])
        b_n = dR_nX.norm2()
        shape = (len(a_n),) + (1,) * (psit_nX.data.ndim - 1)
        lambda_n = (a_n / b_n).reshape(shape)

        # R1_nX = R0_nX + lambda_n * (H - S eps) * P * R0_nX
        dR_nX.data *= lambda_n
        dR_nX.data += R_nX.data
        # Psi1_nX = Psi0_nX + lambda_n * P * R0_nX
        PR_nX.data *= lambda_n
        PR_nX.data += psit_nX.data

        psits_mnX = [psit_nX, PR_nX]
        R_mnX = [R_nX, dR_nX]

        # DIIS Part (only relevant if diis_steps > 1):
        if self.diis_steps > 1:
            blocksize = psit_nX.data.shape[0]

            A_nmm = -xp.ones(
                (blocksize, self.diis_steps + 2, self.diis_steps + 2),
                dtype=as_dtype_precision(dtype, 'double'))
            b_nm = -xp.ones((blocksize, self.diis_steps + 2),
                            dtype=as_dtype_precision(dtype, 'double'))

            for m1, R1_nX in enumerate(R_mnX):
                for m2, R2_nX in enumerate(R_mnX):
                    for b in range(blocksize):
                        A_nmm[b, m1, m2] = R1_nX[b].integrate(R2_nX[b])

            for i in range(1, self.diis_steps + 1):
                # We are already on step 2
                if i < self.diis_steps:
                    # Create some buffers
                    psit_new_nX = psit_nX.create_work_buffer(
                        self.data_buffers[0, i])
                    R_new_nX = psit_nX.create_work_buffer(
                        self.data_buffers[1, i])
                else:
                    # Last step, use original wave functions
                    psit_new_nX = psit_nX
                    R_new_nX = R_nX
                if i > 1:
                    # Avoid DIIS on the first step, since the matrix
                    # is singular.
                    for m in range(i + 1):
                        for b in range(blocksize):
                            A_nmm[b, m, i] = \
                                R_mnX[m][b].integrate(R_mnX[i][b])
                            if m != i:
                                A_nmm[b, i, m] = A_nmm[b, m, i].conj()
                    A_nmm[:, i + 1, i + 1] = 0
                    b_nm[:, :i + 1] = 0
                    lambda_nm = xp.linalg.solve(
                        A_nmm[:, :i + 2, :i + 2],
                        b_nm[:, :i + 2, np.newaxis])[:, :i + 1, 0]

                    psit_new_nX.data[:] = \
                        lambda_nm[:, 0, None]  * psits_mnX[0].data
                    R_new_nX.data[:] = \
                        lambda_nm[:, 0, None] * R_mnX[0].data

                    for m in range(1, i + 1):
                        psit_new_nX.data += \
                            lambda_nm[:, m, None] * psits_mnX[m].data
                        R_new_nX.data += \
                            lambda_nm[:, m, None] * R_mnX[m].data
                else:
                    # Special case for first DIIS step, which is
                    # numerically unstable:
                    psit_new_nX.data[:] = psits_mnX[-1].data
                    R_new_nX.data[:] = R_mnX[-1].data

                if i < self.diis_steps:
                    # Take the step!
                    preconditioner(psit_nX, R_new_nX,
                                   out=R_new_nX, ekin_n=ekin_n)
                    psit_new_nX.data += R_new_nX.data * lambda_n

                    # Get the next trial vector
                    Ht(psit_new_nX, out=R_new_nX)
                    pt_aiX.integrate(psit_new_nX, out=P_ani)  # XXX: Expensive
                    calculate_residuals(psit_new_nX, R_new_nX, pt_aiX,
                                        P_ani, eig_n, dH, dS_aii,
                                        P1_ani, P2_ani)
                R_mnX.append(R_new_nX)
                psits_mnX.append(psit_new_nX)
        else:
            psit_nX.data[:] = psits_mnX[-1].data

        # NUTS Part ("free" bonus diis step):
        preconditioner(psit_nX, R_mnX[-1], out=PR_nX, ekin_n=ekin_n)
        if trial_step is None:
            PR_nX.data *= lambda_n
        else:
            PR_nX.data *= trial_step
        psit_nX.data += PR_nX.data
