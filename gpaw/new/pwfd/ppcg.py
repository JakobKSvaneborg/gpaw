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
from gpaw.new.pwfd.davidson import sliced_preconditioner
# from gpaw.typing import Array2D
from gpaw.core import PWDesc
from gpaw.new import tracectx, trace


class PPCG(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 hamiltonian,
                 converge_bands='occupied',
                 niter=(2, 10),
                 blocksize=None,
                 rr_modulo=5,
                 include_cg=True,
                 promote_inner_dtype=False,
                 tolerances: tuple[float, ...] | None = None,
                 scalapack_parameters=None,
                 max_buffer_mem: int = 200 * 1024 ** 2):
        """
        Initialize Projected Preconditioned Conjugate Gradient eigensolver,
        a.k.a. PPCG or Not-Davidson solver.

        See https://doi.org/10.1016/j.jcp.2015.02.030 for details.

        Parameters
        ----------
        nbands : int
            Number of bands.
        wf_grid : WaveFunctionsGrid
            Grid of wave functions.
        band_comm : MPI communicator
            Communicator for band parallelization.
        hamiltonian : Hamiltonian
            Hamiltonian.
        converge_bands : str, optional
            Which bands to converge ('occupied' or 'unoccupied'). Default is
            'occupied'.
        niter : int | typle[int, int], optional
            Number of iterations. Default is 2. If specified as a tuple,
            the first value is the minimum number of iterations and the
            second value is the maximum number of iterations. Where the
            eigensolver will stop if the residual is below a tolerance,
            and performed at least the minimum number of iterations,
            otherwise the maximum number of iterations is used.
        blocksize : int, optional
            Block size for the diagonal slicing. Lower values
            are more efficient on CPUs with many cores but not on GPUs. The
            value will be modified to a multiple of the number of domain
            ranks.
            Default is 48 on cpu and 128 on gpu.
        rr_modulo : int, optional
            How often to perform subspace diagonalization. Default is 5.
        include_cg : bool, optional
            Include CG in the solver. Default is True. Can be helpful to turn
            off for single precision calculations or if memory is an issue.
        promote_inner_dtype : bool, optional
            Promote inner dtype to double precision. Default is False.
            Only relevant for single precision calculations.
        tolerances : tuple[float, float, float], optional
            Advanced setting, tolerances for the solver. Use at your own risk.
        scalapack_parameters : dict, optional
            Parameters for scalapack solver.
        max_buffer_mem : int, optional
            Maximum memory in bytes for buffer. Default is 200 * 1024 ** 2.
        """

        super().__init__(
            hamiltonian,
            converge_bands)

        if not hamiltonian.band_local:
            raise NotImplementedError(
                'PPCG only implemented for band local XCs,'
                'use davidson instead')

        self.nbands = nbands
        self.wf_grid = wf_grid
        self.band_comm = band_comm
        self.niter = niter
        self.blocksize = blocksize
        self.rr_modulo = rr_modulo
        self.tolerances = tolerances
        self.MW_nn: Matrix
        self.MP_nn: Matrix
        self.include_cg = include_cg
        self.promote_inner_dtype = promote_inner_dtype

    def __str__(self):
        return pformat(dict(name='PPCG',
                            niter=self.niter,
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        xp = ibzwfs.xp

        if self.blocksize is None:
            if xp == np:
                self.blocksize = 48  # Could be lower, maybe 32
            else:
                self.blocksize = 128

        if isinstance(self.wf_grid, PWDesc):
            S = self.wf_grid.comm.size
            # Use a multiple of S for maximum efficiency
            self.blocksize = int(np.ceil(self.blocksize / S)) * S

        super()._initialize(ibzwfs)
        if self.include_cg:
            self._allocate_work_arrays(ibzwfs, shape=(2,))
        else:
            self._allocate_work_arrays(ibzwfs, shape=(1,))
        self._allocate_buffer_arrays(ibzwfs, shape=(1,))

        wfs = ibzwfs.wfs_qs[0][0]
        assert isinstance(wfs, PWFDWaveFunctions)
        band_comm = wfs.band_comm

        B = ibzwfs.nbands
        b = wfs.psit_nX.mydims[0]
        self.blocksize = max(min(self.blocksize, b),
                             1)
        self.nblocksizes = 3 * self.blocksize \
            if self.include_cg else 2 * self.blocksize
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
        self.tol_factor = 0
        # tolerance :
        #   Freeze bands with residual < tolerance
        #   improves numerical stability at the cost of
        #   minimum achievable residual.
        self.tolerance = np.finfo(dtype).eps**2 * G_max**0.5
        # breakout_tolerance :
        #   Stop iteration if sum(residual_ns) < breakout_tolerance
        #   breakout_tolerance saves time at the cost of minimum
        #   achievable residual. Can also be used to improve numerical
        #   stability.
        self.breakout_tolerance = 5e-6

        if self.tolerances is not None:
            assert len(self.tolerances) == 3
            self.tol_factor = self.tolerances[0]
            self.tolerance = self.tolerances[1]
            self.breakout_tolerance = self.tolerances[2]

        self.M_nn = Matrix(B, B, dtype=dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

        if dtype != np.float32 and dtype != np.complex64:
            self.promote_inner_dtype = False

        if self.promote_inner_dtype:
            inner_dtype = np.float64 if np.issubdtype(dtype, np.floating) \
                else np.complex128
            self.buffer_bb = xp.zeros((self.nblocksizes, self.nblocksizes),
                                      dtype=dtype)
        else:
            inner_dtype = dtype

        self.H_bb = xp.zeros((self.nblocksizes, self.nblocksizes),
                             dtype=inner_dtype)
        self.S_bb = xp.zeros((self.nblocksizes, self.nblocksizes),
                             dtype=inner_dtype)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):

        with tracectx('Initialize'):
            M_nn = self.M_nn
            # Some buffer arrays for approx orthonormalization
            # Y1_nn = M_nn.new()
            # Y2_nn = M_nn.new()

            xp = M_nn.xp

            psit_nX = wfs.psit_nX
            b = psit_nX.mydims[0]

            residual_nX = psit_nX.new(data=self.work_arrays[0, :b])
            if self.include_cg:
                P_nX = psit_nX.new(data=self.work_arrays[1, :b])

            wfs.subspace_diagonalize(Ht, dH,
                                     psit2_nX=residual_nX,
                                     data_buffer=self.data_buffers[0])

            P_ani = wfs.P_ani
            P2_ani = P_ani.new()
            P3_ani = P_ani.new()
            Ptemp_ani = P_ani.new()
            P_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
            Pbuf_abi = P_ani.layout.empty((self.nblocksizes, )
                                          + psit_nX.dims[1:])
            HPbuf_abi = Pbuf_abi.new()

            domain_comm = psit_nX.desc.comm
            band_comm = psit_nX.comm

            if weight_n is None:
                weight_n = np.ones(b)

            buffer_array_nX = psit_nX.create_work_buffer(self.data_buffers[0])

            buff_bX = psit_nX.desc.empty((self.nblocksizes, ) +
                                         psit_nX.dims[1:], xp=psit_nX.xp)
            Hbuff_bX = psit_nX.desc.empty((self.nblocksizes, ) +
                                          psit_nX.dims[1:], xp=psit_nX.xp)

            if isinstance(self.niter, int):
                niter = self.niter
                min_niter = self.niter
            else:
                niter = self.niter[1]
                min_niter = self.niter[0]

        with tracectx('Residual'):
            calculate_residuals(wfs.psit_nX,
                                residual_nX,
                                wfs.pt_aiX,
                                wfs.P_ani,
                                wfs.myeig_n,
                                dH, dS_aii, P2_ani, P3_ani)

            error_n = as_np(residual_nX.norm2())
            if len(error_n.shape) > 1:
                error_n = error_n.sum(axis=1)
            active_indicies = np.logical_or(
                np.greater(error_n, self.tolerance),
                np.greater(error_n,
                           np.max(error_n, initial=0) * self.tol_factor))
            active_indicies = np.where(active_indicies)[0]
            error = weight_n @ error_n
            b_error = band_comm.sum_scalar(error) / \
                max(band_comm.sum_scalar(weight_n.sum()), 0.5)
            if band_comm.sum_scalar(len(active_indicies)) == 0  \
                    or b_error < self.breakout_tolerance and min_niter <= 1:
                if debug:
                    psit_nX.sanity_check()
                flag = True
            else:
                flag = False

        for i in range(niter):
            with tracectx('Residual'):
                sliced_preconditioner(psit_nX, residual_nX,
                                      buffer=buffer_array_nX,
                                      precon=self.preconditioner)
                wfs.pt_aiX.integrate(residual_nX, out=P2_ani)
                residual_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                            domain_sum=False,
                                            symmetric=False)
                P2_ani.matrix.multiply(Ptemp_ani, opb='C', symmetric=False,
                                       beta=1, out=M_nn)
                domain_comm.sum(M_nn.data)

                M_nn.multiply(psit_nX, out=residual_nX, beta=1.0, alpha=-1.0)
                M_nn.multiply(P_ani, out=P2_ani, beta=1.0, alpha=-1.0)

            active_bs = len(active_indicies)

            with tracectx('Block-diagonal Update'):
                for j in range(0, active_bs, self.blocksize):
                    block_slice_base = \
                        slice(j, min(j + self.blocksize, active_bs))
                    block = \
                        block_slice_base.stop - block_slice_base.start
                    block_slice = active_indicies[block_slice_base]

                    buff_bX.matrix.data[:block] = \
                        psit_nX.matrix.data[block_slice]
                    Pbuf_abi.matrix.data[:block] = \
                        P_ani.matrix.data[block_slice]
                    buff_bX.matrix.data[block:2 * block] = \
                        residual_nX.matrix.data[block_slice]
                    Pbuf_abi.matrix.data[block:2 * block] = \
                        P2_ani.matrix.data[block_slice]

                    if i > 0 and self.include_cg:
                        nblocks = 3 * block
                        buff_bX.matrix.data[2 * block:3 * block] = \
                            P_nX.matrix.data[block_slice]
                        Pbuf_abi.matrix.data[2 * block:3 * block] = \
                            P3_ani.matrix.data[block_slice]
                    else:
                        nblocks = 2 * block

                    H_bb = self.H_bb.ravel()[:nblocks**2].reshape(
                        (nblocks, nblocks))
                    S_bb = self.S_bb.ravel()[:nblocks**2].reshape(
                        (nblocks, nblocks))

                    MH_bb = Matrix(M=nblocks, N=nblocks,
                                   data=H_bb,
                                   xp=xp)
                    MS_bb = Matrix(M=nblocks, N=nblocks,
                                   data=S_bb,
                                   xp=xp)

                    if self.promote_inner_dtype:
                        buffer_bb = \
                            self.buffer_bb.ravel()[:nblocks**2].reshape(
                                (nblocks, nblocks))
                        MBuf_bb = Matrix(M=nblocks, N=nblocks,
                                         data=buffer_bb,
                                         xp=xp)
                    else:
                        MBuf_bb = MS_bb

                    Pbuf_abi.block_diag_multiply(dS_aii, out_ani=HPbuf_abi)
                    buff_bX[:nblocks].matrix_elements(
                        buff_bX[:nblocks], cc=True, out=MBuf_bb,
                        domain_sum=False, symmetric=True)
                    HPbuf_abi[:, :nblocks].matrix.multiply(
                        Pbuf_abi[:, :nblocks], out=MBuf_bb,
                        symmetric=True, beta=1, opb='C')
                    if self.promote_inner_dtype:
                        S_bb[:] = buffer_bb[:]
                    domain_comm.sum(S_bb)
                    norm_facts = (1 / xp.diag(S_bb)[block:])**0.25
                    S_bb[block:, :] *= norm_facts[:, None]
                    S_bb[:, block:] *= norm_facts[None, :]
                    buff_bX.matrix.data[block:nblocks, :] \
                        *= norm_facts[:, None]
                    Pbuf_abi.matrix.data[block:nblocks, :] \
                        *= norm_facts[:, None]

                    if not self.promote_inner_dtype:
                        MBuf_bb = MH_bb

                    Ht_H = partial(Ht, out=Hbuff_bX[:nblocks])
                    dH(Pbuf_abi[:, :nblocks],
                       out_ani=HPbuf_abi[:, :nblocks])
                    buff_bX[:nblocks].matrix_elements(
                        buff_bX[:nblocks], function=Ht_H,
                        cc=True, out=MBuf_bb,
                        domain_sum=False, symmetric=True)
                    HPbuf_abi[:, :nblocks].matrix.multiply(
                        Pbuf_abi[:, :nblocks], out=MBuf_bb,
                        symmetric=True, beta=1, opb='C')
                    if self.promote_inner_dtype:
                        H_bb[:] = buffer_bb[:]
                    domain_comm.sum(H_bb)

                    if nblocks > 2 * block:
                        # Eigh approach
                        # A, temp_bb = xp.linalg.eigh(S_bb, 'L')
                        # if xp is not np:
                        #     pos_defness = A.get()[0]
                        # Eigvalsh approach
                        with tracectx('eigvalsh', gpu=xp is not np):
                            pos_defness = xp.linalg.eigvalsh(S_bb)[0]
                        if xp is not np:
                            pos_defness = pos_defness.get()
                        if pos_defness < \
                                np.finfo(psit_nX.data.dtype).eps * \
                                nblocks**0.5:
                            # Insufficient numerical precision for CG,
                            # thus we only do the steepest descent step
                            nblocks = 2 * block
                            MH_bb = Matrix(M=nblocks, N=nblocks,
                                           data=H_bb[:nblocks,
                                                     :nblocks],
                                           xp=xp)
                            MS_bb = Matrix(M=nblocks, N=nblocks,
                                           data=S_bb[:nblocks,
                                                     :nblocks],
                                           xp=xp)
                            MH_bb.eigh(MS_bb)
                        else:
                            # Do the full PPCG update
                            MH_bb.eigh(MS_bb)
                    else:
                        MH_bb.eigh(MS_bb)
                    if self.promote_inner_dtype:
                        buffer_bb[:] = H_bb.conj()
                        cmin = buffer_bb[:block, :nblocks]
                    else:
                        cmin = H_bb[:block, :nblocks].conj()
                    if not xp.isfinite(H_bb).all():
                        print('H is not finite')
                        flag = True
                        continue

                    with tracectx('rotations', gpu=xp is not np):
                        # Ye olde updates
                        buff_bX.matrix.data[:block] = \
                            cmin[:, :block] @ buff_bX.matrix.data[:block]
                        Pbuf_abi.matrix.data[:block] = \
                            cmin[:, :block] @ Pbuf_abi.matrix.data[:block]
                        buff_bX.matrix.data[block:2 * block] = \
                            cmin[:, block:] @ buff_bX.matrix.data[
                                block:nblocks]
                        Pbuf_abi.matrix.data[block:2 * block] = \
                            cmin[:, block:] @ Pbuf_abi.matrix.data[
                                block:nblocks]

                        if self.include_cg:
                            P_nX.matrix.data[block_slice] = \
                                buff_bX.matrix.data[block:2 * block]
                            P3_ani.matrix.data[block_slice] = \
                                Pbuf_abi.matrix.data[block:2 * block]

                        psit_nX.matrix.data[block_slice] = \
                            buff_bX.matrix.data[:block] \
                            + buff_bX.matrix.data[block:2 * block]
                        P_ani.matrix.data[block_slice] = \
                            Pbuf_abi.matrix.data[:block] \
                            + Pbuf_abi.matrix.data[block:2 * block]

            wfs.orthonormalized = False
            if flag or i >= niter - 1:
                break

            with tracectx('Residual'):
                # Subspace diagonialization needed every once in a while
                if (i + 1) % self.rr_modulo == 0:
                    # if b_error < 1e-2:
                    #     Approximate orthonormalization only if
                    #     the residual is small.
                    #     approx_orthonormalize(wfs, residual_nX, M_nn, Y1_nn,
                    #                           Y2_nn, dS_aii, domain_comm)
                    wfs.subspace_diagonalize(Ht, dH,
                                             psit2_nX=residual_nX,
                                             data_buffer=self.data_buffers[0])
                else:
                    # In theory we could skip orthonormalization,
                    # but this sometimes causes issues so we do it.
                    if b_error < 1e-2 and False:
                        ...
                        # Approximate orthonormalization only if
                        # the residual is small.
                        # approx_orthonormalize(wfs, residual_nX, M_nn, Y1_nn,
                        #                       Y2_nn, dS_aii, domain_comm)
                    else:
                        wfs.orthonormalize(residual_nX)
                    Ht(psit_nX, out=residual_nX)
                    update_eigenvalues(wfs, residual_nX, P_ani, Ptemp_ani, dH,
                                       domain_comm)

                calculate_residuals(wfs.psit_nX,
                                    residual_nX,
                                    wfs.pt_aiX,
                                    wfs.P_ani,
                                    wfs.myeig_n,
                                    dH, dS_aii, P2_ani, Ptemp_ani)

                error_n = as_np(residual_nX.norm2())
                if len(error_n.shape) > 1:
                    error_n = error_n.sum(axis=1)

                active_indicies = np.logical_or(
                    np.greater(error_n, self.tolerance),
                    np.greater(error_n, np.max(error_n, initial=0) *
                               self.tol_factor))
                active_indicies = np.where(active_indicies)[0]
                error = weight_n @ error_n
                b_error = band_comm.sum_scalar(error) / \
                    max(band_comm.sum_scalar(weight_n.sum()), 0.5)

                if band_comm.sum_scalar(len(active_indicies)) == 0 \
                        or (b_error < self.breakout_tolerance
                            and i + 2 >= min_niter):
                    # Set 'flag = True', causing the loop to break
                    # at the next iteration. This gives us one more
                    # cheap iteration (since we already calculated
                    # the residual).
                    flag = True

            P_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
            if self.include_cg:
                with tracectx('P-update'):
                    P_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                         domain_sum=False,
                                         symmetric=False)
                    P3_ani.matrix.multiply(Ptemp_ani, opb='C',
                                           symmetric=False,
                                           beta=1, out=M_nn)
                    domain_comm.sum(M_nn.data)
                    M_nn.multiply(psit_nX, out=P_nX, beta=1.0, alpha=-1.0)
                    M_nn.multiply(P_ani, out=P3_ani, beta=1.0, alpha=-1.0)

        if not wfs.orthonormalized:
            wfs.orthonormalize(residual_nX)
            # if b_error < 1e-2:
            #     Approximate orthonormalization only if
            #     the residual is small.
            #     approx_orthonormalize(wfs, residual_nX, M_nn, Y1_nn,
            #                           Y2_nn, dS_aii, domain_comm)
            # else:
            #     wfs.orthonormalize(residual_nX)

        if debug:
            psit_nX.sanity_check()

        return error


@trace
def approx_orthonormalize(wfs, residual_nX, Y1_nn, Y2_nn, Y3_nn,
                          dS_aii, domain_comm):
    """
    Approximate orthonormalization of wave functions.

    This function approximates orthonormalization of wave functions
    using a Taylor series expansion of the inverse square root.

    Parameters
    ----------
    wfs : PWFDWaveFunctions
        Wave functions to be orthonormalized.
    residual_nX : Matrix
        Residual matrix to be used as temporary storage.
    Y1_nn, Y2_nn, Y3_nn : Matrix
        Temporary matrices.
    dS_aii : Matrix
        PAW overlap matrix.
    domain_comm : MPI communicator
        Communicator for domain parallelization.
    """
    if wfs.orthonormalized:
        return
    P_ani = wfs.P_ani
    P2_ani = P_ani.new()
    P_ani.block_diag_multiply(dS_aii, out_ani=P2_ani)
    psit_nX = wfs.psit_nX
    psit_nX.matrix_elements(psit_nX, cc=True, out=Y1_nn,
                            domain_sum=False,
                            symmetric=True)
    P_ani.matrix.multiply(P2_ani, opb='C',
                          symmetric=True,
                          beta=1, out=Y1_nn)
    domain_comm.sum(Y1_nn.data)
    Y1_nn.tril2full()

    Y1_nn.add_to_diagonal(-1.0)
    Y1_nn.multiply(Y1_nn, out=Y2_nn)
    Y2_nn.multiply(Y1_nn, out=Y3_nn)
    Y1_nn.data[:] = -(1 / 2) * Y1_nn.data + \
        (3 / 8) * Y2_nn.data + \
        -(5 / 16) * Y3_nn.data

    residual_nX.data[:] = psit_nX.data
    P2_ani.data[:] = P_ani.data

    Y1_nn.multiply(residual_nX, out=psit_nX, beta=1)
    Y1_nn.multiply(P2_ani, out=P_ani, beta=1)
    wfs.orthonormalized = True


@trace
def update_eigenvalues(wfs, Hpsit_nX, P_ani, P2_ani, dH, domain_comm):
    dH(P_ani, out_ani=P2_ani)
    wfs.myeig_n[:] = as_np(
        wfs.psit_nX.approx_eigenvalues(Hpsit_nX, P_ani, P2_ani))
