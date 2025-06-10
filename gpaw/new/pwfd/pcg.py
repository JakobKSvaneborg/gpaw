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
        self.H_NN: Matrix
        self.S_NN: Matrix
        self.M_nn: Matrix

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
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H_NN = Matrix(2 * B, 2 * B, dtype=dtype, xp=xp)
            self.S_NN = Matrix(2 * B, 2 * B, dtype=dtype, xp=xp)
        else:
            self.H_NN = self.S_NN = Matrix(0, 0)

        self.M_nn = Matrix(B, B, dtype=dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        H_NN = self.H_NN
        S_NN = self.S_NN
        M_nn = self.M_nn

        xp = M_nn.xp

        psit_nX = wfs.psit_nX
        B = psit_nX.dims[0]  # number of bands
        b = psit_nX.mydims[0]

        psit2_nX = psit_nX.new(data=self.work_arrays[0, :b])
        psit3_nX = psit_nX.new(data=self.work_arrays[1, :b])

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=psit2_nX.data,
                                 Htpsit_nX=psit3_nX)
        residual_nX = psit3_nX  # will become (H-e*S)|psit> later

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0_nn = M_nn.new(dist=(band_comm, 1, 1))

        Ht = partial(Ht, out=residual_nX)
        Ht(psit_nX, out=residual_nX)
        calculate_residuals(wfs.psit_nX,
                            residual_nX,
                            wfs.pt_aiX,
                            wfs.P_ani,
                            wfs.myeig_n,
                            dH, dS_aii, P2_ani, P3_ani)

        if weight_n is None:
            weight_n = xp.ones(B)
        error_old = (weight_n @ as_np(residual_nX.norm2())).sum()

        P_nX = psit_nX.new()
        blocksize = 4
        C_X = np.zeros((blocksize, blocksize), dtype=complex) # The alphas
        C_W = np.zeros_like(C_X) # The betas
        C_P = np.zeros_like(C_X) # The gammas
        X2_nX = psit_nX.new()
        W2_nX = residual_nX.new()
        P2_nX = P_nX.new()

        flag = False

        for i in range(self.niter):        
            self.preconditioner(psit_nX, residual_nX, out=psit2_nX)
            wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)
            P2_ani.block_diag_multiply(dS_aii, out_ani=P3_ani)

            residual_nX.data[:] = psit2_nX.data
            M_nn = residual_nX.matrix_elements(psit_nX, cc=True)
            P3_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                   out=M_nn)
            M_nn.multiply(psit_nX, out=residual_nX, beta=1.0, alpha=-1.0)
            
            wfs.pt_aiX.integrate(P_nX, out=P2_ani)
            P2_ani.block_diag_multiply(dS_aii, out_ani=P3_ani)
            M_nn = P_nX.matrix_elements(psit_nX, cc=True)
            P3_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                   out=M_nn)
            M_nn.multiply(psit_nX, out=P_nX, beta=1.0, alpha=-1.0)
            
            X = psit_nX.data.T
            W = residual_nX.data.T
            P = P_nX.data.T
            j = 0
            
            wfs.pt_aiX.integrate(residual_nX, out=P2_ani)
            wfs.pt_aiX.integrate(P_nX, out=P3_ani)
            
            Ht(psit_nX, out=X2_nX)
            Ht(residual_nX, out=W2_nX)
            Ht(P_nX, out=P2_nX)
            X2 = X2_nX.data.T
            W2 = W2_nX.data.T
            P2 = P2_nX.data.T
            
            Pbuf1_abi = P_ani.layout.empty(3*blocksize)
            Pbuf2_abi = P_ani.layout.empty(3*blocksize)
            
            while j < b:
                block_slice = slice(j, min(j + blocksize, b))
                # This keeps the block size constant except for the last block
                blocksize = block_slice.stop - block_slice.start

                PX_abi = P_ani[:, block_slice]
                PW_abi = P2_ani[:, block_slice]
                PP_abi = P3_ani[:, block_slice]

                if i > 0:
                    S = np.column_stack([X[:, block_slice], W[:, block_slice], P[:, block_slice]])
                    S2 = np.column_stack([X2[:, block_slice], W2[:, block_slice], P2[:, block_slice]])
                    H_bb = S.T.conj() @ S2 * psit_nX.dv
                    Pbuf1_abi[:, :blocksize].data[:] = PX_abi.data
                    Pbuf1_abi[:, blocksize:2 * blocksize].data[:] = PW_abi.data
                    Pbuf1_abi[:, 2 * blocksize:3 * blocksize].data[:] = PP_abi.data
                    dH(PX_abi, Pbuf2_abi[:, :blocksize])
                    dH(PW_abi, Pbuf2_abi[:, blocksize:2 * blocksize])
                    dH(PP_abi, Pbuf2_abi[:, 2 * blocksize:3 * blocksize])
                    Pbuf1_bx = Pbuf1_abi[:, :3 * blocksize].matrix.data
                    Pbuf2_bx = Pbuf2_abi[:, :3 * blocksize].matrix.data
                    H_bb += Pbuf1_bx.conj() @ Pbuf2_bx.T
                                        
                    M_bb = S.T.conj() @ S * psit_nX.dv
                    PX_abi.block_diag_multiply(dS_aii, out_ani=Pbuf2_abi[:, :blocksize])
                    PW_abi.block_diag_multiply(dS_aii, out_ani=Pbuf2_abi[:, blocksize:2 * blocksize])
                    PP_abi.block_diag_multiply(dS_aii, out_ani=Pbuf2_abi[:, 2 * blocksize:3 * blocksize])
                    Pbuf2_bx = Pbuf2_abi[:, :3 * blocksize].matrix.data
                    M_bb += Pbuf1_bx.conj() @ Pbuf2_bx.T
                else:
                    S = np.column_stack([X[:, block_slice], W[:, block_slice]])
                    S2 = np.column_stack([X2[:, block_slice], W2[:, block_slice]])
                    H_bb = S.T.conj() @ S2 * psit_nX.dv
                    Pbuf1_abi[:, :blocksize].data[:] = PX_abi.data
                    Pbuf1_abi[:, blocksize:2 * blocksize].data[:] = PW_abi.data
                    dH(PX_abi, Pbuf2_abi[:, :blocksize])
                    dH(PW_abi, Pbuf2_abi[:, blocksize:2 * blocksize])
                    Pbuf1_bx = Pbuf1_abi[:, :2 * blocksize].matrix.data
                    Pbuf2_bx = Pbuf2_abi[:, :2 * blocksize].matrix.data
                    H_bb += Pbuf1_bx.conj() @ Pbuf2_bx.T
                                        
                    M_bb = S.T.conj() @ S * psit_nX.dv
                    PX_abi.block_diag_multiply(dS_aii, out_ani=Pbuf2_abi[:, :blocksize])
                    PW_abi.block_diag_multiply(dS_aii, out_ani=Pbuf2_abi[:, blocksize:2 * blocksize])
                    Pbuf2_bx = Pbuf2_abi[:, :2 * blocksize].matrix.data
                    M_bb += Pbuf1_bx.conj() @ Pbuf2_bx.T
                
                # We only need the smallest algebraic eigenvector for this
                # But also this is the tiniest ass eigenvalue problem of 3 blocksize x 3 blocksize...
                _, cmin = sp.linalg.eigh(H_bb, M_bb)
                cmin = cmin[:, :blocksize]
                # Ye olde updates
                C_X[:blocksize, :blocksize] = cmin[:blocksize, :blocksize]
                C_W[:blocksize, :blocksize] = cmin[blocksize:2 * blocksize, :blocksize]
                if i > 0:
                    C_P[:blocksize, :blocksize] = cmin[2 * blocksize:3 * blocksize, :blocksize]
                    P[:, block_slice] = W[:, block_slice] @ C_W[:blocksize, :blocksize] + P[:, block_slice] @ C_P[:blocksize, :blocksize]
                else:
                    P[:, block_slice] = W[:, block_slice] @ C_W[:blocksize, :blocksize]
                X[:, block_slice] = X[:, block_slice] @ C_X[:blocksize, :blocksize] + P[:, block_slice]
                j += blocksize

            psit_nX.data[:] = X.T
            wfs.pt_aiX.integrate(psit_nX, out=P_ani)
            P_nX.data[:] = P.T

            # RR step (only for high niters)
            if (i + 1) % 3 == 0 and i < self.niter - 1:
                wfs.orthonormalized = False
                wfs.orthonormalize(residual_nX.data)

                M2_nn = psit_nX.matrix_elements(psit_nX, function=Ht,
                                                cc=True)
                dH(P_ani, out_ani=P2_ani)
                P_ani.matrix.multiply(P2_ani, opb='C', symmetric=False, beta=1,
                                      out=M2_nn)
                wfs._eig_n[:] = as_np(M2_nn.eigh())
                M2_nn.tril2full()
                M2_nn.multiply(psit_nX, out=psit2_nX)
                psit_nX.data[:] = psit2_nX.data
                M2_nn.multiply(P_ani, out=P3_ani)
                P_ani.data[:] = P3_ani.data

            Ht(psit_nX, out=residual_nX)
            calculate_residuals(wfs.psit_nX,
                                residual_nX,
                                wfs.pt_aiX,
                                wfs.P_ani,
                                wfs.myeig_n,
                                dH, dS_aii, P2_ani, P3_ani)
            error_new = (weight_n @ as_np(residual_nX.norm2())).sum()

            if error_new < 1e-30:
                break
                flag = True
            error_old = error_new
        
        
        wfs.orthonormalized = False
        wfs.orthonormalize(residual_nX.data)
        
        if debug:
            psit_nX.sanity_check()

        error = error_new
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