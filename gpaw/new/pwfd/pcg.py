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
        domain_comm = wfs.psit_nX.desc.comm
        band_comm = wfs.band_comm

        B = ibzwfs.nbands
        xp = ibzwfs.xp
        dtype = wfs.psit_nX.desc.dtype
        self.tolerance = np.finfo(dtype).eps

        self.M_nn = Matrix(B, B, dtype=dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

        self.C_X = xp.zeros((self.blocksize, self.blocksize), dtype=dtype) # The alphas
        self.C_W = xp.zeros_like(self.C_X) # The betas
        self.C_P = xp.zeros_like(self.C_X) # The gammas
        self.H_bb = xp.zeros((3 * self.blocksize, 3 * self.blocksize), dtype=dtype)
        self.S_bb = xp.zeros((3 * self.blocksize, 3 * self.blocksize), dtype=dtype)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
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
        self.preconditioner(psit_nX, residual_nX, out=psit2_nX)
        wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)
        P2_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)

        residual_nX.data[:] = psit2_nX.data
        residual_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                    domain_sum=False)
        Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                  out=M_nn)
        domain_comm.sum(M_nn.data)

        buff_bX = psit_nX.desc.empty(3 * self.blocksize, xp=psit_nX.xp)
        Hbuff_bX = psit_nX.desc.empty(3 * self.blocksize, xp=psit_nX.xp)
        
        active_indicies = np.arange(b)

        for i in range(self.niter):
            M_nn.multiply(psit_nX, out=residual_nX, beta=1.0, alpha=-1.0)
            M_nn.multiply(P_ani, out=P2_ani, beta=1.0, alpha=-1.0)
            
            active_bs = len(active_indicies)
            
            for j in range(0, active_bs, self.blocksize):
                block_slice = slice(j, min(j + self.blocksize, active_bs))
                blocksize = block_slice.stop - block_slice.start
                block_slice = active_indicies[block_slice]
                # This keeps the block size constant except for the last block

                C_X = self.C_X.ravel()[:blocksize**2].reshape(blocksize, blocksize)
                C_W = self.C_W.ravel()[:blocksize**2].reshape(blocksize, blocksize)
                C_P = self.C_P.ravel()[:blocksize**2].reshape(blocksize, blocksize)

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

                H_bb = self.H_bb.ravel()[:nblocksizes**2].reshape(nblocksizes, nblocksizes)
                S_bb = self.S_bb.ravel()[:nblocksizes**2].reshape(nblocksizes, nblocksizes)
                
                MH_bb = Matrix(M=nblocksizes, N=nblocksizes,
                               data=H_bb,
                               xp=xp)
                MS_bb = Matrix(M=nblocksizes, N=nblocksizes,
                               data=S_bb,
                               xp=xp)

                Pbuf_abi.block_diag_multiply(dS_aii, out_ani=HPbuf_abi)
                buff_bX[:nblocksizes].matrix_elements(buff_bX[:nblocksizes], cc=False, out=MS_bb,
                                        domain_sum=False)
                #S_bb[:] = buff_bX.matrix.data[:nblocksizes].conj() @ buff_bX.matrix.data[:nblocksizes].T * psit_nX.dv
                S_bb[:] += Pbuf_abi.matrix.data[:nblocksizes].conj() @ HPbuf_abi.matrix.data[:nblocksizes].T
                domain_comm.sum(S_bb)
                
                Ht(buff_bX[:nblocksizes], out=Hbuff_bX[:nblocksizes])
                dH(Pbuf_abi[:, :nblocksizes], out_ani=HPbuf_abi[:, :nblocksizes])
                Hbuff_bX[:nblocksizes].matrix_elements(buff_bX[:nblocksizes], cc=False, out=MH_bb,
                                         domain_sum=False)
                #HPbuf_abi.matrix.multiply(Pbuf_abi, opb='C', symmetric=False, beta=1,
                #                          out=MH_bb)
                #H_bb[:] = buff_bX.matrix.data[:nblocksizes].conj() @ Hbuff_bX.matrix.data[:nblocksizes].T * psit_nX.dv
                H_bb[:] += Pbuf_abi.matrix.data[:nblocksizes].conj() @ HPbuf_abi.matrix.data[:nblocksizes].T
                domain_comm.sum(H_bb)

                # We only need the smallest algebraic eigenvector for this
                # But also this is the tiniest ass eigenvalue problem of 3 * blocksize x 3 * blocksize...
                MH_bb.eigh(MS_bb)
                cmin = H_bb[:blocksize, :]
                # Ye olde updates
                C_X[:] = cmin[:blocksize, :blocksize]
                C_W[:] = cmin[:blocksize, blocksize:2 * blocksize]
                if i > 0:
                    C_P[:] = cmin[:blocksize, 2 * blocksize:3 * blocksize]
                    P_nX.matrix.data[block_slice] = C_W @ buff_bX.matrix.data[blocksize:2 * blocksize] \
                        + C_P @ buff_bX.matrix.data[2*blocksize:3 * blocksize]
                    P3_ani.matrix.data[block_slice] = C_W @ Pbuf_abi.matrix.data[blocksize:2 * blocksize] \
                        + C_P @ Pbuf_abi.matrix.data[2*blocksize:3 * blocksize]
                else:
                    P_nX.matrix.data[block_slice] = C_W @ buff_bX.matrix.data[blocksize:2 * blocksize]
                    P3_ani.matrix.data[block_slice] = C_W @ Pbuf_abi.matrix.data[blocksize:2 * blocksize]
                psit_nX.matrix.data[block_slice] = C_X @ buff_bX.matrix.data[:blocksize] \
                    + P_nX.matrix.data[block_slice]
                P_ani.matrix.data[block_slice] = C_X @ Pbuf_abi.matrix.data[:blocksize] \
                    + P3_ani.matrix.data[block_slice]

            wfs.orthonormalized = False

            # Subspace diagonialization needed every once in a while
            if (i + 1) % 5 == 0 :
                wfs.subspace_diagonalize(Ht, dH,
                                         work_array=residual_nX.data)

            Ht(psit_nX, out=residual_nX)
            calculate_residuals(wfs.psit_nX,
                                residual_nX,
                                wfs.pt_aiX,
                                wfs.P_ani,
                                wfs.myeig_n,
                                dH, dS_aii, P2_ani, Ptemp_ani)
            
            error_ns = as_np(residual_nX.norm2())
            active_indicies = np.where(np.greater(error_ns, self.tolerance))[0]
            error = (weight_n @ error_ns).sum()
            b_error = band_comm.sum_scalar(error)
            
            if len(active_indicies) == 0:
                print(f'Converged in {i + 1} iterations')
                break
            
            if i < self.niter - 1:
                P3_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
                P_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                     domain_sum=False)
                Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                          out=M_nn)
                domain_comm.sum(M_nn.data)
                M_nn.multiply(psit_nX, out=P_nX, beta=1.0, alpha=-1.0)
                M_nn.multiply(P_ani, out=P3_ani, beta=1.0, alpha=-1.0)

                self.preconditioner(psit_nX, residual_nX, out=residual_nX)
                wfs.pt_aiX.integrate(residual_nX, out=P2_ani)
                P2_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
                residual_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                            domain_sum=False)
                Ptemp_ani.matrix.multiply(P_ani, opb='C', symmetric=False, beta=1,
                                          out=M_nn)
                domain_comm.sum(M_nn.data)
        
        #if not wfs.orthonormalized:
            #wfs.subspace_diagonalize(Ht, dH,
            #                         work_array=residual_nX.data)
            #wfs.orthonormalize(residual_nX.data)
        
        if debug:
            psit_nX.sanity_check()

        return error
