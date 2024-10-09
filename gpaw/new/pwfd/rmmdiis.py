from __future__ import annotations

from functools import partial
from pprint import pformat

import numpy as np

from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
from gpaw.mpi import broadcast_exception
from gpaw.new import trace, zips
from gpaw.new.pwfd.eigensolver import PWFDEigensolver
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D


class RMMDIIS(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 niter=2,
                 blocksize=10,
                 converge_bands='occupied',
                 scalapack_parameters=None):
        self.niter = niter
        self.converge_bands = converge_bands

        self.H_NN = None
        self.S_NN = None
        self.M_nn = None
        self.work_arrays: np.ndarray | None = None

        self.preconditioner = None
        self.preconditioner_factory = preconditioner_factory
        self.blocksize = blocksize

        ...
        if self.blocksize is None:
            if wfs.mode == 'pw':
                S = wfs.pd.comm.size
                # Use a multiple of S for maximum efficiency
                self.blocksize = int(np.ceil(10 / S)) * S
            else:
                self.blocksize = 10

    def __str__(self):
        return pformat(dict(name='Davidson',
                            niter=self.niter,
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        # First time: allocate work-arrays
        wfs = ibzwfs.wfs_qs[0][0]
        assert isinstance(wfs, PWFDWaveFunctions)
        xp = wfs.psit_nX.xp
        self.preconditioner = self.preconditioner_factory(self.blocksize,
                                                          xp=xp)
        B = ibzwfs.nbands
        b = max(wfs.n2 - wfs.n1 for wfs in ibzwfs)
        domain_comm = wfs.psit_nX.desc.comm
        band_comm = wfs.band_comm
        shape = ibzwfs.get_max_shape()
        shape = (2, b) + shape
        dtype = wfs.psit_nX.data.dtype
        self.work_arrays = xp.empty(shape, dtype)

        dtype = wfs.psit_nX.desc.dtype
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H_NN = Matrix(2 * B, 2 * B, dtype, xp=xp)
            self.S_NN = Matrix(2 * B, 2 * B, dtype, xp=xp)
        else:
            self.H_NN = self.S_NN = Matrix(0, 0)

        self.M_nn = Matrix(B, B, dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

    @trace
    def iterate(self,
                ibzwfs,
                density,
                potential,
                hamiltonian: Hamiltonian) -> float:
        """Iterate on state given fixed hamiltonian.

        Returns
        -------
        float:
            Weighted error of residuals:::

                   ~     ~ ~
              R = (H - ε S)ψ
               n        n   n
        """

        if self.work_arrays is None:
            self._initialize(ibzwfs)

        assert self.M_nn is not None

        wfs = ibzwfs.wfs_qs[0][0]
        dS_aii = wfs.setups.get_overlap_corrections(wfs.P_ani.layout.atomdist,
                                                    wfs.xp)
        dH = potential.dH
        Ht = partial(hamiltonian.apply,
                     potential.vt_sR,
                     potential.dedtaut_sR,
                     ibzwfs, density.D_asii)  # used by hybrids

        weight_un = calculate_weights(self.converge_bands, ibzwfs)

        error = 0.0
        with broadcast_exception(ibzwfs.kpt_comm):
            for wfs, weight_n in zips(ibzwfs, weight_un):
                e = self.iterate1(wfs, Ht, dH, dS_aii, weight_n)
                error += wfs.weight * e
        return ibzwfs.kpt_band_comm.sum_scalar(
            float(error)) * ibzwfs.spin_degeneracy

    @trace
    def iterate1(self, wfs, Ht, dH, dS_aii, weight_n):
        self.subspace_diagonalize(ham, wfs, kpt)

        psit = kpt.psit
        # psit2 = psit.new(buf=wfs.work_array)
        P = kpt.projections
        P2 = P.new()
        # dMP = P.new()
        # M_nn = wfs.work_matrix_nn
        # dS = wfs.setups.dS
        R = psit.new(buf=self.Htpsit_nG)

        self.calculate_residuals(kpt, wfs, ham, psit, P, kpt.eps_n,
                                 R, P2)

        def integrate(a_G, b_G):
            return np.real(wfs.integrate(a_G, b_G, global_integral=False))

        comm = wfs.gd.comm

        B = self.blocksize
        dR = R.new(dist=None, nbands=B)
        dpsit = dR.new()
        P = P.new(bcomm=None, nbands=B)
        P2 = P.new()
        errors_x = np.zeros(B)

        Ht = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

        error = 0.0
        for n1 in range(0, wfs.bd.mynbands, B):
            n2 = n1 + B
            if n2 > wfs.bd.mynbands:
                n2 = wfs.bd.mynbands
                B = n2 - n1
                P = P.new(nbands=B)
                P2 = P.new()
                dR = dR.new(nbands=B, dist=None)
                dpsit = dR.new()

            n_x = np.arange(n1, n2)
            psitb = psit.view(n1, n2)

            Rb = R.view(n1, n2)

            errors_x[:] = 0.0
            for n in range(n1, n2):
                weight = weights[n]
                errors_x[n - n1] = weight * integrate(Rb.array[n - n1],
                                                      Rb.array[n - n1])
            comm.sum(errors_x)
            error += np.sum(errors_x)

            # Precondition the residual:
            ekin_x = self.preconditioner.calculate_kinetic_energy(
                psitb.array, kpt)
            self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)

            # Calculate the residual of dpsit_G, dR_G = (H - e S) dpsit_G:
            # self.timer.start('Apply Hamiltonian')
            dpsit.apply(Ht, out=dR)
            # self.timer.stop('Apply Hamiltonian')
            dpsit.matrix_elements(wfs.pt, out=P)

            self.calculate_residuals(kpt, wfs, ham, dpsit,
                                     P, kpt.eps_n[n_x], dR, P2, n_x,
                                     calculate_change=True)

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            RdR_x = np.array([integrate(dR_G, R_G)
                              for R_G, dR_G in zip(Rb.array, dR.array)])
            dRdR_x = np.array([integrate(dR_G, dR_G) for dR_G in dR.array])
            comm.sum(RdR_x)
            comm.sum(dRdR_x)
            lam_x = -RdR_x / dRdR_x

            # New trial wavefunction and residual
            for lam, psit_G, dpsit_G, R_G, dR_G in zip(
                lam_x, psitb.array,
                dpsit.array, Rb.array,
                dR.array):
            axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
            axpy(lam, dR_G, R_G)  # R_G += lam** dR_G

            # Final trial step
            self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)

            for lam, psit_G, dpsit_G in zip(lam_x, psitb.array, dpsit.array):
                axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
