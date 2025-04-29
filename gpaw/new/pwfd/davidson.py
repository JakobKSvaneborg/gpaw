from __future__ import annotations

from pprint import pformat

import numpy as np
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
from gpaw.mpi import broadcast_exception
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D
from gpaw.new import trace, tracectx


class Davidson(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 hamiltonian,
                 converge_bands='occupied',
                 niter=2,
                 scalapack_parameters=None,
                 max_buffer_mem: int = 200 * 1024 ** 2):
        super().__init__(
            hamiltonian,
            converge_bands,
            max_buffer_mem=max_buffer_mem)
        self.niter = niter
        self.H_NN: Matrix
        self.S_NN: Matrix
        self.M_nn: Matrix

    def __str__(self):
        return pformat(dict(name='Davidson',
                            niter=self.niter,
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        self._allocate_work_arrays(ibzwfs, shape=(1,))
        self._allocate_buffer_arrays(ibzwfs, shape=(1,))

        wfs = ibzwfs.wfs_qs[0][0]
        assert isinstance(wfs, PWFDWaveFunctions)
        domain_comm = wfs.psit_nX.desc.comm
        band_comm = wfs.band_comm

        B = ibzwfs.nbands
        xp = ibzwfs.xp
        dtype = wfs.psit_nX.desc.dtype
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H_NN = Matrix(2 * B, 2 * B, dtype, xp=xp)
            self.S_NN = Matrix(2 * B, 2 * B, dtype, xp=xp)
        else:
            self.H_NN = self.S_NN = Matrix(0, 0)

        self.M_nn = Matrix(B, B, dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        H_NN = self.H_NN
        S_NN = self.S_NN
        M_nn = self.M_nn
        M2_nn = M_nn.new()

        xp = M_nn.xp

        psit_nX = wfs.psit_nX
        B = psit_nX.dims[0]  # number of bands
        eig_N = xp.empty(2 * B)
        b = psit_nX.mydims[0]

        psit2_nX = psit_nX.new(data=self.work_arrays[0, :b])
        data_buffer = self.data_buffers[0]

        wfs.subspace_diagonalize(Ht, dH,
                                 psit2_nX=psit2_nX,
                                 data_buffer=data_buffer)

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0_nn = M_nn.new(dist=(band_comm, 1, 1))

        if domain_comm.rank == 0:
            eig_N[:B] = xp.asarray(wfs.eig_n)

        me_buffer_mX = psit_nX.new_buffer(data_buffer)

        @trace
        def me(a, b, function=None, sliced=False):
            """Matrix elements"""
            return a.matrix_elements(b,
                                     domain_sum=False,
                                     out=M_nn,
                                     function=function,
                                     cc=True)

        calculate_residuals(wfs.psit_nX,
                            psit2_nX,
                            wfs.pt_aiX,
                            wfs.P_ani,
                            wfs.myeig_n,
                            dH, dS_aii, P2_ani, P3_ani)

        def copy(C_nn: Array2D, M_nn: Matrix) -> None:
            domain_comm.sum(M_nn.data, 0)
            if domain_comm.rank == 0:
                M_nn.redist(M0_nn)
                if band_comm.rank == 0:
                    C_nn[:] = M0_nn.data

        for i in range(self.niter):
            if i == self.niter - 1:  # last iteration
                # Calculate error before we destroy residuals:
                if weight_n is None:
                    error = np.inf
                else:
                    error = (weight_n @ as_np(psit2_nX.norm2())).sum()

            sliced_preconditioner(psit_nX, psit2_nX,
                                  buffer=me_buffer_mX,
                                  precon=self.preconditioner)

            # Calculate projections
            wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)
            with tracectx('Matrix elements'):
                sliced_matrix_elements(psit_nX, psit2_nX,
                                       buffer_mX=me_buffer_mX,
                                       Ht=Ht,
                                       M1_nn=M_nn,
                                       M2_nn=M2_nn)

                # <psi2 | H | psi2>
                dH(P2_ani, out_ani=P3_ani)
                P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                       out=M2_nn)
                copy(H_NN.data[B:, B:], M2_nn)

                # <psi2 | H | psi>
                P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
                copy(H_NN.data[B:, :B], M_nn)

                # <psi2 | S | psi2>
                me(psit2_nX, psit2_nX)
                P2_ani.block_diag_multiply(dS_aii, out_ani=P3_ani)
                P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                       out=M_nn)
                copy(S_NN.data[B:, B:], M_nn)

                # <psi2 | S | psi>
                me(psit2_nX, psit_nX)
                P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
                copy(S_NN.data[B:, :B], M_nn)

            with tracectx('Diagonalize'):
                with broadcast_exception(domain_comm):
                    with broadcast_exception(band_comm):
                        if is_domain_band_master:
                            H_NN.data[:B, :B] = xp.diag(eig_N[:B])
                            S_NN.data[:B, :B] = xp.eye(B)
                            eig_N[:] = H_NN.eigh(S_NN)
                            wfs._eig_n = as_np(eig_N[:B])
                if domain_comm.rank == 0:
                    band_comm.broadcast(wfs.eig_n, 0)
                domain_comm.broadcast(wfs.eig_n, 0)

                if domain_comm.rank == 0:
                    if band_comm.rank == 0:
                        M0_nn.data[:] = H_NN.data[:B, :B]
                        M0_nn.complex_conjugate()
                    M0_nn.redist(M_nn)
                domain_comm.broadcast(M_nn.data, 0)

            with tracectx('Rotate Psi'):
                M_nn.multiply(psit_nX, out=psit_nX,
                              data_buffer=data_buffer)
                M_nn.multiply(P_ani, out=P3_ani)

                if domain_comm.rank == 0:
                    if band_comm.rank == 0:
                        M0_nn.data[:] = H_NN.data[:B, B:]
                        M0_nn.complex_conjugate()
                    M0_nn.redist(M_nn)
                domain_comm.broadcast(M_nn.data, 0)

                M_nn.multiply(psit2_nX, beta=1.0, out=psit_nX)
                M_nn.multiply(P2_ani, beta=1.0, out=P3_ani)
                P_ani, P3_ani = P3_ani, P_ani
                wfs._P_ani = P_ani

            if i < self.niter - 1:
                Ht(psit_nX, out=psit2_nX)
                calculate_residuals(
                    wfs.psit_nX,
                    psit2_nX,
                    wfs.pt_aiX, wfs.P_ani, wfs.myeig_n,
                    dH, dS_aii, P2_ani, P3_ani)

        if debug:
            psit_nX.sanity_check()

        return error


def sliced_preconditioner(psit_nX, psit2_nX, buffer, precon):
    # Sliced recursive preconditioning
    buffer_size = buffer.data.shape[0]
    mybands = psit_nX.data.shape[0]
    if not mybands == 0:
        for i_local in range(0, mybands, buffer_size):
            buffer_view = buffer[:mybands - i_local]
            precon(
                psit_nX[i_local:i_local + buffer_size],
                psit2_nX[i_local:i_local + buffer_size],
                out=buffer_view)
            psit2_nX.data[i_local:i_local + buffer_size] \
                = buffer_view.data[:]


def sliced_matrix_elements(psit1_nX, psit2_nX, buffer_mX, Ht, M1_nn, M2_nn):
    comm = psit1_nX.comm
    b = psit1_nX.data.shape[0]
    blocksize = buffer_mX.data.shape[0]
    blocksize_world = comm.sum_scalar(blocksize)
    totalbands = comm.sum_scalar(b)
    for i1, N1 in enumerate(
            range(0, totalbands, blocksize_world)):
        n1 = i1 * blocksize
        n2 = n1 + blocksize
        if n2 > b:
            n2 = b

        Ht(psit2_nX[n1:n2], out=buffer_mX[:n2 - n1])
        M1 = psit1_nX.matrix
        M2 = psit2_nX.matrix

        buffer_view_matrix = Matrix(
            M=min(blocksize_world,
                  totalbands - N1),
            N=M2.data.shape[1],
            data=buffer_mX[:n2 - n1].matrix.data,
            dist=(comm, -1, 1),
            xp=M2.xp)
        out1 = Matrix(
            M=min(blocksize_world,
                  totalbands - N1),
            N=M1_nn.shape[1],
            data=M1_nn.data[n1:n2, :],
            dist=(comm, -1, 1),
            xp=M1_nn.xp)
        out2 = Matrix(
            M=min(blocksize_world,
                  totalbands - N1),
            N=M2_nn.shape[1],
            data=M2_nn.data[n1:n2, :],
            dist=(comm, -1, 1),
            xp=M2_nn.xp)

        buffer_view_matrix.multiply(M1, alpha=psit1_nX.dv, opb='C',
                                    out=out1)
        buffer_view_matrix.multiply(M2, alpha=psit2_nX.dv, opb='C',
                                    out=out2)
        psit1_nX._matrix_elements_correction(buffer_view_matrix, M1, out1,
                                             symmetric=False)
        psit2_nX._matrix_elements_correction(buffer_view_matrix, M2, out2,
                                             symmetric=False)
