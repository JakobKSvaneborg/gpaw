from __future__ import annotations

from functools import partial
from pprint import pformat

import numpy as np
import cupy as cp
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
# from gpaw.mpi import broadcast_exception
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.pwfd.davidson import sliced_preconditioner
# from gpaw.typing import Array2D
from gpaw.new import tracectx, trace


class NotDavidson(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 hamiltonian,
                 converge_bands='occupied',
                 niter=4,
                 blocksize=128,
                 rr_modulo=5,
                 include_CG=True,
                 tolerances: tuple[float] | None = None,
                 scalapack_parameters=None,
                 max_buffer_mem: int = 200 * 1024 ** 2):
        """
        Initialize Projected Preconditioned Conjugate Gradient eigensolver.
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
        niter : int, optional
            Number of iterations. Default is 4.
        blocksize : int, optional
            Block size for the diagonal slicing. Default is 256, lower values
            are more efficient on CPUs with many cores but not on GPUs. The
            value will be modified to a multiple of the number of domain
            ranks.
        rr_modulo : int, optional
            How often to perform subspace diagonalization. Default is 5.
        include_CG : bool, optional
            Include CG in the solver. Default is True. Can be helpfull to turn
            off for single precision calculations or if memory is an issue.
        tolerances : tuple[float], optional
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
                'NotDavidson only implemented for band local XCs')

        self.nbands = nbands
        self.wf_grid = wf_grid
        self.band_comm = band_comm
        self.niter = niter
        self.blocksize = blocksize
        self.rr_modulo = rr_modulo
        self.nblocksizes = 3 * self.blocksize if include_CG \
            else 2 * self.blocksize
        self.tolerances = tolerances
        self.MW_nn: Matrix
        self.MP_nn: Matrix
        self.include_CG = include_CG

    def __str__(self):
        return pformat(dict(name='Not Davidson',
                            niter=self.niter,
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        if self.include_CG:
            self._allocate_work_arrays(ibzwfs, shape=(2,))
        else:
            self._allocate_work_arrays(ibzwfs, shape=(1,))
        self._allocate_buffer_arrays(ibzwfs, shape=(1,))

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
        self.tolerance = np.finfo(dtype).eps**2 * G_max
        # breakout_tolerance :
        #   Stop iteration if sum(residual_ns) < breakout_tolerance
        #   breakout_tolerance saves time at the cost of minimum
        #   achievable residual. Can also be used to improve numerical
        #   stability.
        self.breakout_tolerance = \
            np.finfo(dtype).eps**2 * (
                B * extra_dims * G_max)
        # initial_tolerance :
        #   Only do subspace diagonalization if
        #   sum(residual_ns) < initial_breakout_tolerance
        #   This value can be lower, since the first iteration
        #   is more numerically stable.
        self.initial_tolerance_factor = 0

        if self.tolerances is not None:
            assert len(self.tolerances) == 4
            self.tol_factor = self.tolerances[0]
            self.tolerance = self.tolerances[1]
            self.breakout_tolerance = self.tolerances[2]
            self.initial_tolerance_factor = self.tolerances[3]

        self.M_nn = Matrix(B, B, dtype=dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

        self.H_bb = xp.zeros((self.nblocksizes, self.nblocksizes),
                             dtype=dtype)
        self.S_bb = xp.zeros((self.nblocksizes, self.nblocksizes),
                             dtype=dtype)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):

        with tracectx('Initialize'):
            M_nn = self.M_nn
            Y1_nn = M_nn.new()
            Y2_nn = M_nn.new()

            xp = M_nn.xp

            psit_nX = wfs.psit_nX
            b = psit_nX.mydims[0]

            residual_nX = psit_nX.new(data=self.work_arrays[0, :b])
            if self.include_CG:
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

            Ht = partial(Ht, out=residual_nX)
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
                    or b_error < self.breakout_tolerance * \
                    self.initial_tolerance_factor:
                print('No active bands.')
                return error

            buffer_array_nX = psit_nX.create_work_buffer(self.data_buffers[0])

            buff_bX = psit_nX.desc.empty((self.nblocksizes, ) +
                                         psit_nX.dims[1:], xp=psit_nX.xp)
            Hbuff_bX = psit_nX.desc.empty((self.nblocksizes, ) +
                                          psit_nX.dims[1:], xp=psit_nX.xp)

        flag = False

        for i in range(self.niter):
            with tracectx('Residual'):
                sliced_preconditioner(psit_nX, residual_nX,
                                      buffer=buffer_array_nX,
                                      precon=self.preconditioner)
                wfs.pt_aiX.integrate(residual_nX, out=P2_ani)
                residual_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                            domain_sum=False)
                P2_ani.matrix.multiply(Ptemp_ani, opb='C', symmetric=False,
                                       beta=1, out=M_nn)
                domain_comm.sum(M_nn.data)

                M_nn.multiply(psit_nX, out=residual_nX, beta=1.0, alpha=-1.0)
                M_nn.multiply(P_ani, out=P2_ani, beta=1.0, alpha=-1.0)

            active_bs = len(active_indicies)

            with tracectx('Blockdiagonal Update'):
                for j in range(0, active_bs, self.blocksize):
                    block_slice_base = \
                        slice(j, min(j + self.blocksize, active_bs))
                    blocksize = \
                        block_slice_base.stop - block_slice_base.start
                    block_slice = active_indicies[block_slice_base]

                    buff_bX.matrix.data[:blocksize] = \
                        psit_nX.matrix.data[block_slice]
                    Pbuf_abi.matrix.data[:blocksize] = \
                        P_ani.matrix.data[block_slice]
                    buff_bX.matrix.data[blocksize:2 * blocksize] = \
                        residual_nX.matrix.data[block_slice]
                    Pbuf_abi.matrix.data[blocksize:2 * blocksize] = \
                        P2_ani.matrix.data[block_slice]

                    if i > 0 and self.include_CG:
                        nblocksizes = 3 * blocksize
                        buff_bX.matrix.data[2 * blocksize:3 * blocksize] = \
                            P_nX.matrix.data[block_slice]
                        Pbuf_abi.matrix.data[2 * blocksize:3 * blocksize] = \
                            P3_ani.matrix.data[block_slice]
                    else:
                        nblocksizes = 2 * blocksize

                    H_bb = self.H_bb.ravel()[:nblocksizes**2].reshape(
                        (nblocksizes, nblocksizes))
                    S_bb = self.S_bb.ravel()[:nblocksizes**2].reshape(
                        (nblocksizes, nblocksizes))

                    MH_bb = Matrix(M=nblocksizes, N=nblocksizes,
                                   data=H_bb,
                                   xp=xp)
                    MS_bb = Matrix(M=nblocksizes, N=nblocksizes,
                                   data=S_bb,
                                   xp=xp)

                    Pbuf_abi.block_diag_multiply(dS_aii, out_ani=HPbuf_abi)
                    buff_bX[:nblocksizes].matrix_elements(
                        buff_bX[:nblocksizes], cc=True, out=MS_bb,
                        domain_sum=False)
                    S_bb[:] += Pbuf_abi.matrix.data[:nblocksizes] @ \
                        HPbuf_abi.matrix.data[:nblocksizes].T.conj()
                    domain_comm.sum(S_bb)

                    Ht(buff_bX[:nblocksizes], out=Hbuff_bX[:nblocksizes])
                    dH(Pbuf_abi[:, :nblocksizes],
                       out_ani=HPbuf_abi[:, :nblocksizes])
                    buff_bX[:nblocksizes].matrix_elements(
                        Hbuff_bX[:nblocksizes], cc=True, out=MH_bb,
                        domain_sum=False)
                    H_bb[:] += Pbuf_abi.matrix.data[:nblocksizes] @ \
                        HPbuf_abi.matrix.data[:nblocksizes].T.conj()
                    domain_comm.sum(H_bb)

                    if nblocksizes > 2 * blocksize:
                        pos_defness = xp.linalg.eigh(S_bb)[0][0]
                        if xp is cp:
                            pos_defness = pos_defness.get()
                        if pos_defness < \
                                np.finfo(S_bb.dtype).eps * nblocksizes**0.5:
                            nblocksizes = 2 * blocksize
                            MH_bb = Matrix(M=nblocksizes, N=nblocksizes,
                                           data=H_bb[:nblocksizes,
                                                     :nblocksizes],
                                           xp=xp)
                            MS_bb = Matrix(M=nblocksizes, N=nblocksizes,
                                           data=S_bb[:nblocksizes,
                                                     :nblocksizes],
                                           xp=xp)
                    MH_bb.eigh(MS_bb)
                    cmin = H_bb[:blocksize, :nblocksizes].conj()
                    if not xp.isfinite(H_bb).all():
                        flag = True
                        continue

                    # Ye olde updates
                    buff_bX.matrix.data[:blocksize] = \
                        cmin[:, :blocksize] @ buff_bX.matrix.data[:blocksize]
                    Pbuf_abi.matrix.data[:blocksize] = \
                        cmin[:, :blocksize] @ Pbuf_abi.matrix.data[:blocksize]
                    buff_bX.matrix.data[blocksize:2 * blocksize] = \
                        cmin[:, blocksize:] @ buff_bX.matrix.data[
                            blocksize:nblocksizes]
                    Pbuf_abi.matrix.data[blocksize:2 * blocksize] = \
                        cmin[:, blocksize:] @ Pbuf_abi.matrix.data[
                            blocksize:nblocksizes]

                    if self.include_CG:
                        P_nX.matrix.data[block_slice] = \
                            buff_bX.matrix.data[blocksize:2 * blocksize]
                        P3_ani.matrix.data[block_slice] = \
                            Pbuf_abi.matrix.data[blocksize:2 * blocksize]

                    psit_nX.matrix.data[block_slice] = \
                        buff_bX.matrix.data[:blocksize] \
                        + buff_bX.matrix.data[blocksize:2 * blocksize]
                    P_ani.matrix.data[block_slice] = \
                        Pbuf_abi.matrix.data[:blocksize] \
                        + Pbuf_abi.matrix.data[blocksize:2 * blocksize]

            wfs.orthonormalized = False
            if flag:
                print('Encountered singular matrix in iteration %d' % i)
                break

            with tracectx('Residual'):
                # Subspace diagonialization needed every once in a while
                if (i + 1) % self.rr_modulo == 0:
                    wfs.subspace_diagonalize(Ht, dH,
                                             psit2_nX=residual_nX,
                                             data_buffer=self.data_buffers[0])
                else:
                    if xp.isnan(psit_nX.data).any():
                        raise ValueError('NaN in wave function')
                    if b_error < 0:
                        approx_orthonormalize(wfs, residual_nX, M_nn, Y1_nn,
                                              Y2_nn, dS_aii, domain_comm)
                    else:
                        wfs.orthonormalize(residual_nX)
                    if xp.isnan(psit_nX.data).any():
                        raise ValueError('NaN in wave function')
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
                P_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
                if self.include_CG:
                    with tracectx('P-update'):
                        P_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                             domain_sum=False)
                        P3_ani.matrix.multiply(Ptemp_ani, opb='C',
                                               symmetric=False,
                                               beta=1, out=M_nn)
                        domain_comm.sum(M_nn.data)
                        M_nn.multiply(psit_nX, out=P_nX, beta=1.0, alpha=-1.0)
                        M_nn.multiply(P_ani, out=P3_ani, beta=1.0, alpha=-1.0)

        if xp.isnan(psit_nX.data).any():
            raise ValueError('NaN in wave function')

        if not wfs.orthonormalized:
            wfs.orthonormalize(residual_nX)

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
                            domain_sum=False)
    P_ani.matrix.multiply(P2_ani, opb='C',
                          symmetric=False,
                          beta=1, out=Y1_nn)
    domain_comm.sum(Y1_nn.data)

    Y1_nn.add_to_diagonal(-1.0)
    Y1_nn.multiply(Y1_nn, out=Y2_nn)
    Y2_nn.multiply(Y1_nn, out=Y3_nn)
    Y1_nn.data[:] = -(1 / 2) * Y1_nn.data[:] + \
        (3 / 8) * Y2_nn.data[:] + \
        -(5 / 16) * Y3_nn.data[:]

    residual_nX.data[:] = psit_nX.data[:]
    P2_ani.data[:] = P_ani.data[:]

    Y1_nn.multiply(residual_nX, out=psit_nX, beta=1)
    Y1_nn.multiply(P2_ani, out=P_ani, beta=1)
    wfs.orthonormalized = True
