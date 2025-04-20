from __future__ import annotations

from pprint import pformat

import numpy as np
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
from gpaw.mpi import broadcast_exception
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D, Array1D
from gpaw.new import trace, tracectx
from gpaw.utilities import as_complex_dtype

MAX_MEM = int(2e8)  # ~200 MB, seems to be the sweet spot


class Davidson(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 converge_bands='occupied',
                 niter=2,
                 scalapack_parameters=None):
        super().__init__(
            preconditioner_factory,
            converge_bands)
        self.niter = niter
        self.H_NN: Matrix
        self.S_NN: Matrix
        self.M_nn: Matrix
        self.data_buffer: Array1D | None = None

    def __str__(self):
        return pformat(dict(name='Davidson',
                            niter=self.niter,
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        self._allocate_work_arrays(ibzwfs, shape=(1,))

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

        G_max = np.prod(ibzwfs.get_max_shape())
        psit_nX = wfs.psit_nX.matrix
        mybands = psit_nX.shape[0]
        complex_dtype = as_complex_dtype(dtype)

        # Single buffer approach
        buffer_size = max(min(MAX_MEM,
                              psit_nX.data.shape[0] * G_max
                              * complex_dtype.itemsize),
                          G_max * complex_dtype.itemsize,
                          2 * mybands * complex_dtype.itemsize)
        self.data_buffer = xp.empty((buffer_size,), np.byte)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        H_NN = self.H_NN
        S_NN = self.S_NN
        M_nn = self.M_nn

        xp = M_nn.xp

        psit_nX = wfs.psit_nX
        B = psit_nX.dims[0]  # number of bands
        eig_N = xp.empty(2 * B)
        b = psit_nX.mydims[0]

        psit2_nX = psit_nX.new(data=self.work_arrays[0, :b])

        wfs.subspace_diagonalize(Ht, dH,
                                 psit2_nX=psit2_nX,
                                 data_buffer=self.data_buffer)

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0_nn = M_nn.new(dist=(band_comm, 1, 1))

        if domain_comm.rank == 0:
            eig_N[:B] = xp.asarray(wfs.eig_n)

        me_buffer_mX = psit_nX.new_buffer(self.data_buffer)
        
        
        @trace
        def me(a, b, function=None, sliced=False):
            """Matrix elements"""
            return a.matrix_elements(b,
                                     domain_sum=False,
                                     out=M_nn,
                                     function=function,
                                     cc=True,
                                     sliced=sliced,
                                     buffer=me_buffer_mX)

        calculate_residuals(wfs.psit_nX,
                            psit2_nX,
                            wfs.pt_aiX,
                            wfs.P_ani,
                            wfs.myeig_n,
                            dH, dS_aii, P2_ani, P3_ani)

        def copy(C_nn: Array2D) -> None:
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

            # Sliced preconditioning
            buffer_size = me_buffer_mX.data.shape[0]
            mybands = psit_nX.data.shape[0]
            if not mybands == 0:
                for i_local in range(0, mybands, buffer_size):
                    buffer_view = me_buffer_mX[:mybands - i_local]
                    self.preconditioner(psit_nX[i_local:i_local + buffer_size],
                                        psit2_nX[i_local:i_local + buffer_size],
                                        buffer_view)
                    psit2_nX.data[i_local:i_local + buffer_size] = buffer_view.data[:]
            
            # self.preconditioner(psit_nX, psit2_nX, psit2_nX)
            # Calculate projections
            wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)
            with tracectx('Matrix elements'):
                # <psi2 | H | psi2>
                #psit3_nX = psit2_nX.new()
                #from functools import partial
                #me(psit2_nX, psit2_nX, function=partial(Ht, out=psit3_nX))
                me(psit2_nX, psit2_nX, function=Ht, sliced=True)
                dH(P2_ani, out_ani=P3_ani)
                P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                       out=M_nn)
                copy(H_NN.data[B:, B:])

                # <psi2 | H | psi>
                #me(psit3_nX, psit_nX)
                me(psit2_nX, psit_nX, function=Ht, sliced=True)
                P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
                copy(H_NN.data[B:, :B])

                # <psi2 | S | psi2>
                me(psit2_nX, psit2_nX)
                P2_ani.block_diag_multiply(dS_aii, out_ani=P3_ani)
                P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                       out=M_nn)
                copy(S_NN.data[B:, B:])

                # <psi2 | S | psi>
                me(psit2_nX, psit_nX)
                P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
                copy(S_NN.data[B:, :B])

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
                              data_buffer=self.data_buffer)
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
