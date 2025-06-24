from __future__ import annotations

from functools import partial
from pprint import pformat

import numpy as np
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
# from gpaw.mpi import broadcast_exception
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
# from gpaw.typing import Array2D
# from gpaw.new import trace, tracectx


class NotDavidson(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 hamiltonian,
                 converge_bands='occupied',
                 niter=4,
                 scalapack_parameters=None,
                 max_buffer_mem: int = 200 * 1024 ** 2):
        super().__init__(
            hamiltonian,
            converge_bands)

        if not hamiltonian.band_local:
            raise NotImplementedError(
                'NotDavidson only implemented for band local XCs')

        self.nbands = nbands
        self.wf_grid = wf_grid
        self.band_comm = band_comm
        self.niter = niter
        self.blocksize = 100
        self.MW_nn: Matrix
        self.MP_nn: Matrix
        self.tolerance: float

    def __str__(self):
        return pformat(dict(name='Not Davidson',
                            niter=self.niter,
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        self._allocate_work_arrays(ibzwfs, shape=(2,))

        wfs = ibzwfs.wfs_qs[0][0]
        assert isinstance(wfs, PWFDWaveFunctions)
        band_comm = wfs.band_comm

        B = ibzwfs.nbands
        extra_dims = np.prod(wfs.psit_nX.dims[1:])
        xp = ibzwfs.xp
        dtype = wfs.psit_nX.desc.dtype
        G_max = np.prod(ibzwfs.get_max_shape())

        # --------------- Convergence parameters ---------------
        # Mostly relevant for single precision, however the
        # breakout_tolerance could be used to speed up convergence
        # in double precision.
        #
        # tol_factor :
        #   Freeze bands with residual < tol_factor * max(residual_ns).
        #   improves numerical stability at the cost of
        #   convergence speed - up to a certain point.
        #   Probably best to not use this one.
        self.tol_factor = 0  # np.finfo(dtype).eps
        # tolerance :
        #   Freeze bands with residual < tolerance
        #   improves numerical stability at the cost of
        #   minimum achievable residual.
        self.tolerance = 5e5 * np.finfo(dtype).eps**2 * np.sqrt(G_max)
        # breakout_tolerance :
        #   Stop iteration if sum(residual_ns) < breakout_tolerance
        #   breakout_tolerance saves time at the cost of minimum
        #   achievable residual. Can also be used to improve numerical
        #   stability.
        self.breakout_tolerance = \
            5e5 * np.finfo(dtype).eps**2 * np.sqrt(
                B * extra_dims * G_max)
        # initial_tolerance :
        #   Only do subspace diagonalization if
        #   sum(residual_ns) < initial_breakout_tolerance
        #   This value can be lower, since the first iteration
        #   is more numerically stable.
        self.initial_tolerance_factor = self.tolerance

        self.M_nn = Matrix(B, B, dtype=dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

        self.C_X = xp.zeros((self.blocksize, self.blocksize),
                            dtype=dtype)  # The alphas
        self.C_W = self.C_X.copy()  # The betas
        self.C_P = self.C_X.copy()  # The gammas
        self.H_bb = xp.zeros((3 * self.blocksize, 3 * self.blocksize),
                             dtype=dtype)
        self.S_bb = xp.zeros((3 * self.blocksize, 3 * self.blocksize),
                             dtype=dtype)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        M_nn = self.M_nn

        xp = M_nn.xp

        psit_nX = wfs.psit_nX
        b = psit_nX.mydims[0]

        psit2_nX = psit_nX.new(data=self.work_arrays[0, :b])
        psit3_nX = psit_nX.new(data=self.work_arrays[1, :b])

        wfs.subspace_diagonalize(Ht, dH,
                                 psit2_nX=psit2_nX)
        residual_nX = psit3_nX  # will become (H-e*S)|psit> later
        P_nX = psit2_nX

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()
        Ptemp_ani = P_ani.new()
        Pbuf_abi = P_ani.layout.empty((3 * self.blocksize, )
                                      + psit_nX.dims[1:])
        HPbuf_abi = P_ani.layout.empty((3 * self.blocksize, )
                                       + psit_nX.dims[1:])

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm

        if weight_n is None:
            weight_n = np.ones(b)

        Ht = partial(Ht, out=residual_nX)
        Ht(psit_nX, out=residual_nX)
        calculate_residuals(wfs.psit_nX,
                            residual_nX,
                            wfs.pt_aiX,
                            wfs.P_ani,
                            wfs.myeig_n,
                            dH, dS_aii, P2_ani, P3_ani)

        error_n = as_np(residual_nX.norm2())
        if len(error_n.shape) > 1:
            error_n = error_n.sum(axis=1)
        active_indicies = np.logical_and(
            np.greater(error_n,
                       self.initial_tolerance_factor * self.tolerance),
            np.greater(error_n,
                       np.max(error_n, initial=0) * self.tol_factor))
        active_indicies = np.where(active_indicies)[0]
        error = weight_n @ error_n
        b_error = band_comm.sum_scalar(error)
        if band_comm.sum_scalar(len(active_indicies)) == 0  \
                or b_error < self.breakout_tolerance \
                * self.initial_tolerance_factor:
            print('No active bands.')
            return error

        self.preconditioner(psit_nX, residual_nX, out=psit2_nX)
        wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)
        P2_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)

        residual_nX.data[:] = psit2_nX.data
        residual_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                    domain_sum=False)
        Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                  out=M_nn)
        domain_comm.sum(M_nn.data)

        buff_bX = psit_nX.desc.empty((3 * self.blocksize, ) +
                                     psit_nX.dims[1:], xp=psit_nX.xp)
        Hbuff_bX = psit_nX.desc.empty((3 * self.blocksize, ) +
                                      psit_nX.dims[1:], xp=psit_nX.xp)

        for i in range(self.niter):
            M_nn.multiply(psit_nX, out=residual_nX, beta=1.0, alpha=-1.0)
            M_nn.multiply(P_ani, out=P2_ani, beta=1.0, alpha=-1.0)

            active_bs = len(active_indicies)

            for j in range(0, active_bs, self.blocksize):
                block_slice_base = \
                    slice(j, min(j + self.blocksize, active_bs))
                blocksize = \
                    block_slice_base.stop - block_slice_base.start
                block_slice = active_indicies[block_slice_base]

                C_X = self.C_X.ravel()[:blocksize**2].reshape(
                    (blocksize, blocksize))
                C_W = self.C_W.ravel()[:blocksize**2].reshape(
                    (blocksize, blocksize))
                C_P = self.C_P.ravel()[:blocksize**2].reshape(
                    (blocksize, blocksize))

                buff_bX.matrix.data[:blocksize] = \
                    psit_nX.matrix.data[block_slice]
                Pbuf_abi.matrix.data[:blocksize] = \
                    P_ani.matrix.data[block_slice]
                buff_bX.matrix.data[blocksize:2 * blocksize] = \
                    residual_nX.matrix.data[block_slice]
                Pbuf_abi.matrix.data[blocksize:2 * blocksize] = \
                    P2_ani.matrix.data[block_slice]

                if i > 0:
                    nblocksizes = 3 * blocksize
                    buff_bX.matrix.data[2 * blocksize:3 * blocksize] = \
                        P_nX.matrix.data[block_slice]
                    Pbuf_abi.matrix.data[2 * blocksize:3 * blocksize] = \
                        P3_ani.matrix.data[block_slice]

                else:
                    nblocksizes = 2 * blocksize

                H_bb = self.H_bb.ravel()[:nblocksizes**2].reshape(
                    nblocksizes, nblocksizes)
                S_bb = self.S_bb.ravel()[:nblocksizes**2].reshape(
                    nblocksizes, nblocksizes)

                MH_bb = Matrix(M=nblocksizes, N=nblocksizes,
                               data=H_bb,
                               xp=xp)
                MS_bb = Matrix(M=nblocksizes, N=nblocksizes,
                               data=S_bb,
                               xp=xp)

                Pbuf_abi.block_diag_multiply(dS_aii, out_ani=HPbuf_abi)
                buff_bX[:nblocksizes].matrix_elements(
                    buff_bX[:nblocksizes], cc=False, out=MS_bb,
                    domain_sum=False)
                S_bb[:] += Pbuf_abi.matrix.data[:nblocksizes].conj() @ \
                    HPbuf_abi.matrix.data[:nblocksizes].T
                domain_comm.sum(S_bb)

                buff_bX[:nblocksizes].matrix_elements(
                    buff_bX[:nblocksizes], cc=False, out=MS_bb,
                    domain_sum=False)
                S_bb[:] += Pbuf_abi.matrix.data[:nblocksizes].conj() @ \
                    HPbuf_abi.matrix.data[:nblocksizes].T
                domain_comm.sum(S_bb)

                Ht(buff_bX[:nblocksizes], out=Hbuff_bX[:nblocksizes])
                dH(Pbuf_abi[:, :nblocksizes],
                   out_ani=HPbuf_abi[:, :nblocksizes])
                Hbuff_bX[:nblocksizes].matrix_elements(
                    buff_bX[:nblocksizes], cc=False, out=MH_bb,
                    domain_sum=False)
                H_bb[:] += Pbuf_abi.matrix.data[:nblocksizes].conj() @ \
                    HPbuf_abi.matrix.data[:nblocksizes].T
                domain_comm.sum(H_bb)

                MH_bb.eigh(MS_bb)
                cmin = H_bb[:blocksize, :]
                # Ye olde updates
                C_X[:] = cmin[:blocksize, :blocksize]
                C_W[:] = cmin[:blocksize, blocksize:2 * blocksize]
                if nblocksizes == 3 * blocksize:
                    C_P[:] = cmin[:blocksize, 2 * blocksize:3 * blocksize]
                    P_nX.matrix.data[block_slice] = \
                        C_W @ buff_bX.matrix.data[blocksize:2 * blocksize] + \
                        C_P @ buff_bX.matrix.data[2 * blocksize:3 * blocksize]
                    P3_ani.matrix.data[block_slice] = \
                        C_W @ Pbuf_abi.matrix.data[blocksize:2 * blocksize] + \
                        C_P @ Pbuf_abi.matrix.data[2 * blocksize:3 * blocksize]
                else:
                    P_nX.matrix.data[block_slice] = \
                        C_W @ buff_bX.matrix.data[blocksize:2 * blocksize]
                    P3_ani.matrix.data[block_slice] = \
                        C_W @ Pbuf_abi.matrix.data[blocksize:2 * blocksize]
                psit_nX.matrix.data[block_slice] = \
                    C_X @ buff_bX.matrix.data[:blocksize] \
                    + P_nX.matrix.data[block_slice]
                P_ani.matrix.data[block_slice] = \
                    C_X @ Pbuf_abi.matrix.data[:blocksize] \
                    + P3_ani.matrix.data[block_slice]

            wfs.orthonormalized = False

            # Subspace diagonialization needed every once in a while
            if (i + 1) % 5 == 0:
                wfs.subspace_diagonalize(Ht, dH,
                                         psit2_nX=residual_nX)

            Ht(psit_nX, out=residual_nX)
            calculate_residuals(wfs.psit_nX,
                                residual_nX,
                                wfs.pt_aiX,
                                wfs.P_ani,
                                wfs.myeig_n,
                                dH, dS_aii, P2_ani, Ptemp_ani)

            error_n = as_np(residual_nX.norm2())
            if len(error_n.shape) > 1:
                error_n = error_n.sum(axis=1)
            active_indicies = np.logical_and(
                np.greater(error_n, self.tolerance),
                np.greater(error_n, np.max(error_n, initial=0) *
                           self.tol_factor))
            active_indicies = np.where(active_indicies)[0]
            error = weight_n @ error_n
            b_error = band_comm.sum_scalar(error)

            if band_comm.sum_scalar(len(active_indicies)) == 0 \
                    or b_error < self.breakout_tolerance:
                print(f'Converged in {i + 1} iterations')
                break

            if i < self.niter - 1:
                P3_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
                P_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                     domain_sum=False)
                Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False,
                                          beta=1, out=M_nn)
                domain_comm.sum(M_nn.data)
                M_nn.multiply(psit_nX, out=P_nX, beta=1.0, alpha=-1.0)
                M_nn.multiply(P_ani, out=P3_ani, beta=1.0, alpha=-1.0)

                self.preconditioner(psit_nX, residual_nX, out=residual_nX)
                wfs.pt_aiX.integrate(residual_nX, out=P2_ani)
                P2_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
                residual_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                            domain_sum=False)
                Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False,
                                          beta=1, out=M_nn)
                domain_comm.sum(M_nn.data)

        if not wfs.orthonormalized:
            wfs.orthonormalize(residual_nX)

        if debug:
            psit_nX.sanity_check()

        return error
