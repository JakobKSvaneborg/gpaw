from __future__ import annotations

from functools import partial
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

MAX_MEM = 2e5 # ~20 MB

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
        self.H_NN = None
        self.S_NN = None
        self.M_nn = None
        self.work_arrays: np.ndarray | None = None
        self.buffers_nx: List[Matrix] | None = None
        self.buffer_mX: np.ndarray | None = None

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
        
        G_max = ibzwfs.get_max_shape()[0]
        psit_nX = wfs.psit_nX.matrix
        dist = psit_nX.dist
        
        # allocate buffers for Ht @ psit
        buffer_size = max(
            min(int(dist.comm.size * MAX_MEM /
                    (max(B, 1) *
                         dtype.itemsize)),
                G_max), 1)
        buffer_nx = Matrix(M=B, N=buffer_size,
                           dtype=dtype,
                           dist=(dist.new(M=psit_nX.shape[0],
                                          N=buffer_size)),
                            xp=psit_nX.xp)
        print(buffer_nx.shape)
        self.buffers_nx = [buffer_nx, buffer_nx.new()]
        
        # preallocate buffer for psit @ Ht(psit). XXX: Is there a way to obtain one-buffer-fits-all?
        [wfs.psit_nX.get_buffer(max_mem=MAX_MEM) for wfs in ibzwfs]

    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        H_NN = self.H_NN
        S_NN = self.S_NN
        M_nn = self.M_nn

        xp = M_nn.xp

        psit_nX = wfs.psit_nX
        B = psit_nX.dims[0]  # number of bands
        eig_N = xp.empty(2 * B)
        b = psit_nX.mydims[0]

        psit2_nX = psit_nX.new(data=self.work_arrays[0, :b])
        #psit3_nX = psit_nX.new(data=self.work_arrays[1, :b])

        wfs.subspace_diagonalize(Ht, dH,
                                 psit2_nX=psit2_nX,
                                 buffer_nx=self.buffers_nx)

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0_nn = M_nn.new(dist=(band_comm, 1, 1))

        if domain_comm.rank == 0:
            eig_N[:B] = xp.asarray(wfs.eig_n)

        @trace
        def me(a, b, function=None, sliced=False):
            """Matrix elements"""
            return a.matrix_elements(b,
                                     domain_sum=False,
                                     out=M_nn,
                                     function=function,
                                     cc=True,
                                     sliced=sliced,
                                     buffer=self.buffer_mX)

        #Ht = partial(Ht, out=residual_nX)#, spin=wfs.spin)
        #dH = partial(dH, spin=wfs.spin)
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
                    error = weight_n @ as_np(psit2_nX.norm2())
                    if wfs.ncomponents == 4:
                        error = error.sum()

            self.preconditioner(psit_nX, psit2_nX, out=psit2_nX)

            # Calculate projections
            wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)

            with tracectx('Matrix elements'):
                # <psi2 | H | psi2>
                me(psit2_nX, psit2_nX, function=Ht, sliced=True)
                dH(P2_ani, out_ani=P3_ani)
                P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                       out=M_nn)
                copy(H_NN.data[B:, B:])

                # <psi2 | H | psi>
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
                M_nn.multiply(psit_nX, out=psit_nX)
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
                partial(Ht, out=psit2_nX)(psit_nX)
                calculate_residuals(
                    wfs.psit_nX,
                    psit2_nX,
                    wfs.pt_aiX, wfs.P_ani, wfs.myeig_n,
                    dH, dS_aii, P2_ani, P3_ani)

        if debug:
            psit_nX.sanity_check()

        return error
