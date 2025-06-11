from __future__ import annotations

from functools import partial
from pprint import pformat

import numpy as np
import scipy as sp
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
from gpaw.mpi import broadcast_exception
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D
from gpaw.new import trace, tracectx


class NotDavidson(PWFDEigensolver):
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
        self.blocksize = 32
        self.MW_nn: Matrix
        self.MP_nn: Matrix

    def __str__(self):
        return pformat(dict(name='Davidson',
                            niter=self.niter,
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        self._allocate_work_arrays(ibzwfs, shape=(2,))

        wfs = ibzwfs.wfs_qs[0][0]
        assert isinstance(wfs, PWFDWaveFunctions)
        domain_comm = wfs.psit_nX.desc.comm
        band_comm = wfs.band_comm

        B = ibzwfs.nbands
        xp = ibzwfs.xp
        dtype = wfs.psit_nX.desc.dtype

        self.MW_nn = Matrix(B, B, dtype=dtype,
                            dist=(band_comm, band_comm.size),
                            xp=xp)
        
        self.MP_nn = Matrix(B, B, dtype=dtype,
                            dist=(band_comm, band_comm.size),
                            xp=xp)

        self.C_X = xp.zeros((self.blocksize, self.blocksize), dtype=complex) # The alphas
        self.C_W = xp.zeros_like(self.C_X) # The betas
        self.C_P = xp.zeros_like(self.C_X) # The gammas
        self.H_bb = xp.zeros((3 * self.blocksize, 3 * self.blocksize), dtype=complex)
        self.S_bb = xp.zeros((3 * self.blocksize, 3 * self.blocksize), dtype=complex)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        MW_nn = self.MW_nn
        MP_nn = self.MP_nn

        xp = MW_nn.xp

        psit_nX = wfs.psit_nX
        B = psit_nX.dims[0]  # number of bands
        b = psit_nX.mydims[0]

        psit2_nX = psit_nX.new(data=self.work_arrays[0, :b])
        psit3_nX = psit_nX.new(data=self.work_arrays[1, :b])

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=psit2_nX.data,
                                 Htpsit_nX=psit3_nX)
        residual_nX = psit3_nX  # will become (H-e*S)|psit> later
        P_nX = psit2_nX

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()
        Ptemp_ani = P_ani.new()
        Pbuf_abi = P_ani.layout.empty(3 * self.blocksize)
        HPbuf_abi = P_ani.layout.empty(3 * self.blocksize)

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        Ht = partial(Ht, out=residual_nX)
        Ht(psit_nX, out=residual_nX)
        calculate_residuals(wfs.psit_nX,
                            residual_nX,
                            wfs.pt_aiX,
                            wfs.P_ani,
                            wfs.myeig_n,
                            dH, dS_aii, P2_ani, P3_ani)
        self.preconditioner(psit_nX, residual_nX, out=psit2_nX)
        wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)
        P2_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)

        residual_nX.data[:] = psit2_nX.data
        residual_nX.matrix_elements(psit_nX, cc=True, out=MW_nn,
                                    domain_sum=False)
        Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                  out=MW_nn)
        domain_comm.sum(MW_nn.data)

        buff_bX = psit_nX.desc.empty(3 * self.blocksize, xp=psit_nX.xp)
        Hbuff_bX = psit_nX.desc.empty(3 * self.blocksize, xp=psit_nX.xp)

        for i in range(self.niter):
            MW_nn.multiply(psit_nX, out=residual_nX, beta=1.0, alpha=-1.0)
            MW_nn.multiply(P_ani, out=P2_ani, beta=1.0, alpha=-1.0)
            
            for j in range(0, b, self.blocksize):
                block_slice = slice(j, min(j + self.blocksize, b))
                # This keeps the block size constant except for the last block
                blocksize = block_slice.stop - block_slice.start

                C_X = self.C_X.ravel()[:blocksize*blocksize].reshape(blocksize, blocksize)
                C_W = self.C_W.ravel()[:blocksize*blocksize].reshape(blocksize, blocksize)
                C_P = self.C_P.ravel()[:blocksize*blocksize].reshape(blocksize, blocksize)

                buff_bX.data[:blocksize] = psit_nX.data[block_slice]
                Pbuf_abi.matrix.data[:blocksize] = P_ani.matrix.data[block_slice]
                buff_bX.data[blocksize:2 * blocksize] = residual_nX.data[block_slice]
                Pbuf_abi.matrix.data[blocksize:2 * blocksize] = P2_ani.matrix.data[block_slice]

                if i > 0:
                    nblocksizes = 3 * blocksize
                    buff_bX.data[2*blocksize:3 * blocksize] = P_nX.data[block_slice]
                    Pbuf_abi.matrix.data[2*blocksize:3 * blocksize] = P3_ani.matrix.data[block_slice]
                else:
                    nblocksizes = 2 * blocksize

                H_bb = self.H_bb.ravel()[:nblocksizes*nblocksizes].reshape(nblocksizes, nblocksizes)
                S_bb = self.S_bb.ravel()[:nblocksizes*nblocksizes].reshape(nblocksizes, nblocksizes)

                Pbuf_abi.block_diag_multiply(dS_aii, out_ani=HPbuf_abi)
                S_bb[:] = buff_bX.matrix.data[:nblocksizes].conj() @ buff_bX.matrix.data[:nblocksizes].T * psit_nX.dv
                S_bb[:] += Pbuf_abi.matrix.data[:nblocksizes].conj() @ HPbuf_abi.matrix.data[:nblocksizes].T
                domain_comm.sum(S_bb)
                
                Ht(buff_bX[:nblocksizes], out=Hbuff_bX[:nblocksizes])
                dH(Pbuf_abi[:, :nblocksizes], out_ani=HPbuf_abi[:, :nblocksizes])
                H_bb[:] = buff_bX.matrix.data[:nblocksizes].conj() @ Hbuff_bX.matrix.data[:nblocksizes].T * psit_nX.dv
                H_bb[:] += Pbuf_abi.matrix.data[:nblocksizes].conj() @ HPbuf_abi.matrix.data[:nblocksizes].T
                domain_comm.sum(H_bb)

                # We only need the smallest algebraic eigenvector for this
                # But also this is the tiniest ass eigenvalue problem of 3 * blocksize x 3 * blocksize...
                if xp is np:
                    _, cmin = sp.linalg.eigh(H_bb, S_bb)
                else:
                    _, cmin = xp.linalg.eigh(H_bb, S_bb)
                cmin = cmin[:, :blocksize]  # ... Transpose?
                # Ye olde updates
                C_X[:] = cmin[:blocksize, :blocksize].T
                C_W[:] = cmin[blocksize:2 * blocksize, :blocksize].T
                if i > 0:
                    C_P[:] = cmin[2 * blocksize:3 * blocksize, :blocksize].T
                    P_nX.data[block_slice] = C_W @ buff_bX.data[blocksize:2 * blocksize] \
                        + C_P @ buff_bX.data[2*blocksize:3 * blocksize]
                    P3_ani.matrix.data[block_slice] = C_W @ Pbuf_abi.matrix.data[blocksize:2 * blocksize] \
                        + C_P @ Pbuf_abi.matrix.data[2*blocksize:3 * blocksize]
                else:
                    P_nX.data[block_slice] = C_W @ buff_bX.data[blocksize:2 * blocksize]
                    P3_ani.matrix.data[block_slice] = C_W @ Pbuf_abi.matrix.data[blocksize:2 * blocksize]
                psit_nX.data[block_slice] = C_X @ buff_bX.data[:blocksize] \
                    + P_nX.data[block_slice]
                P_ani.matrix.data[block_slice] = C_X @ Pbuf_abi.matrix.data[:blocksize] \
                    + P3_ani.matrix.data[block_slice]

            wfs.orthonormalized = False

            # Subspace diagonialization needed every once in a while
            if (i + 1) % 3 == 0:
                wfs.subspace_diagonalize(Ht, dH,
                                         work_array=residual_nX.data)

            Ht(psit_nX, out=residual_nX)
            calculate_residuals(wfs.psit_nX,
                                residual_nX,
                                wfs.pt_aiX,
                                wfs.P_ani,
                                wfs.myeig_n,
                                dH, dS_aii, P2_ani, Ptemp_ani)
            
            if i < self.niter - 1:
                P3_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
                P_nX.matrix_elements(psit_nX, cc=True, out=MP_nn,
                                     domain_sum=False)
                Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                          out=MP_nn)
                domain_comm.sum(MP_nn.data)
                MP_nn.multiply(psit_nX, out=P_nX, beta=1.0, alpha=-1.0)
                MP_nn.multiply(P_ani, out=P3_ani, beta=1.0, alpha=-1.0)

                self.preconditioner(psit_nX, residual_nX, out=residual_nX)
                wfs.pt_aiX.integrate(residual_nX, out=P2_ani)
                P2_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
                residual_nX.matrix_elements(psit_nX, cc=True, out=MW_nn,
                                            domain_sum=False)
                Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                          out=MW_nn)
                domain_comm.sum(MW_nn.data)

        if weight_n is None:
            error = np.inf
        else:
            error = (weight_n @ as_np(residual_nX.norm2())).sum()

        wfs.orthonormalize(residual_nX.data)
        
        if debug:
            psit_nX.sanity_check()

        return error

'''
@profile
def ppcg(A, k=6, T=None, X=None, blocksize=60, rr_interval=5, qr_interval=5):
    
    if T is None:
        # vals, vecs = np.linalg.eigh(A)
        # T = np.linalg.inv(A)
        # T = np.eye(len(A))#np.diag(np.diag(A))
        T = np.diag(1 / np.diag(A))
    n = len(A)
    if X is None:
        X, _ = np.linalg.qr(np.random.rand(len(A), k))

    # Strictly speaking, for the first iteration, this shouldn't exist...
    P = np.random.rand(n, k) * 1e-8

    C_X = np.zeros((k, k)) # The alphas
    C_W = np.zeros_like(C_X) # The betas
    C_P = np.zeros_like(C_X) # The gammas
    traceold = np.sum(np.diag(X.T @ A @ X))
    for iconvergence in range(500):
        AX = A @ X # H psi
        # XXT = X @ X.T
        W = T @ (AX - X @ (X.T @ AX))
        W -= X @ (X.T @ W)
        P -= X @ (X.T @ P) if P is not None else P
        j = 0
        while j < k:
            block_slice = slice(j, min(j + blocksize, k))
            # This keeps the block size constant except for the last block
            blocksize = block_slice.stop - block_slice.start

            S = np.column_stack([X[:, block_slice], W[:, block_slice], P[:, block_slice]])
            # We only need the smallest algebraic eigenvector for this
            # But also this is the tiniest ass eigenvalue problem of 3 blocksize x 3 blocksize...
            thetamin, cmin = eigh(S.T @ A @ S, S.T @ S)
            thetamin = thetamin[0]
            cmin = cmin[:, :blocksize]
            # Ye olde updates
            C_X[block_slice, block_slice] = cmin[:blocksize, :blocksize]
            C_W[block_slice, block_slice] = cmin[blocksize:2 * blocksize, :blocksize]
            C_P[block_slice, block_slice] = cmin[2 * blocksize:3 * blocksize, :blocksize] if iconvergence != 0 else 0.0
            P[:, block_slice] = W[:, block_slice] @ C_W[:blocksize, block_slice]+ P[:, block_slice] @ C_P[block_slice, block_slice]
            X[:, block_slice] = X[:, block_slice] @ C_X[block_slice, block_slice]+ P[:, block_slice]
            j += blocksize
        # RR step?
        if iconvergence % qr_interval == 0:
            X, _ = np.linalg.qr(X)
        if iconvergence % rr_interval == 0:
            vals, vecs = eigh(X.T @ A @ X)
            X = X @ vecs

            tracenew = np.sum(np.diag(X.T @ A @ X))
            convergence_marker = np.abs(tracenew - traceold) / traceold
            print(iconvergence, convergence_marker)
            if convergence_marker < 1e-8:
                break
            traceold = tracenew

    return vals, X
'''