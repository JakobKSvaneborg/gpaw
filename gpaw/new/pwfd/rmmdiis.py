from functools import partial

import numpy as np

from gpaw.utilities.blas import axpy
from gpaw.eigensolvers.eigensolver import Eigensolver


class RMMDIIS(Eigensolver):
    """RMM-DIIS eigensolver

    It is expected that the trial wave functions are orthonormal
    and the integrals of projector functions and wave functions
    ``nucleus.P_uni`` are already calculated

    Solution steps are:

    * Subspace diagonalization
    * Calculation of residuals
    * Improvement of wave functions:  psi' = psi + lambda PR + lambda PR'
    * Orthonormalization"""

    def __init__(self, rtol=1e-16):

        Eigensolver.__init__(self, keep_htpsit, blocksize)
        self.niter = niter
        self.rtol = rtol
        self.limit_lambda = limit_lambda
        self.use_rayleigh = use_rayleigh
        if use_rayleigh:
            1 / 0
            self.blocksize = 1
        self.trial_step = trial_step
        self.first = True

    def todict(self):
        return {'name': 'rmm-diis', 'niter': self.niter}

    def initialize(self, wfs):
        if self.blocksize is None:
            if wfs.mode == 'pw':
                S = wfs.pd.comm.size
                # Use a multiple of S for maximum efficiency
                self.blocksize = int(np.ceil(10 / S)) * S
            else:
                self.blocksize = 10
        Eigensolver.initialize(self, wfs)

    def iterate_one_k_point(self, ham, wfs, kpt, weights):
        """Do a single RMM-DIIS iteration for the kpoint"""

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

            with self.timer('Calculate residuals'):
                Rb = R.view(n1, n2)

            errors_x[:] = 0.0
            for n in range(n1, n2):
                weight = weights[n]
                errors_x[n - n1] = weight * integrate(Rb.array[n - n1],
                                                      Rb.array[n - n1])
            comm.sum(errors_x)
            error += np.sum(errors_x)

            # Precondition the residual:
            with self.timer('precondition'):
                ekin_x = self.preconditioner.calculate_kinetic_energy(
                    psitb.array, kpt)
                self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)

            # Calculate the residual of dpsit_G, dR_G = (H - e S) dpsit_G:
            # self.timer.start('Apply Hamiltonian')
            dpsit.apply(Ht, out=dR)
            # self.timer.stop('Apply Hamiltonian')
            with self.timer('projections'):
                dpsit.matrix_elements(wfs.pt, out=P)

            with self.timer('Calculate residuals'):
                self.calculate_residuals(kpt, wfs, ham, dpsit,
                                         P, kpt.eps_n[n_x], dR, P2, n_x,
                                         calculate_change=True)

            # Find lam that minimizes the norm of R'_G = R_G + lam dR_G
            with self.timer('Find lambda'):
                RdR_x = np.array([integrate(dR_G, R_G)
                                  for R_G, dR_G in zip(Rb.array, dR.array)])
                dRdR_x = np.array([integrate(dR_G, dR_G) for dR_G in dR.array])
                comm.sum(RdR_x)
                comm.sum(dRdR_x)
                lam_x = -RdR_x / dRdR_x

            # New trial wavefunction and residual
            with self.timer('Update psi'):
                for lam, psit_G, dpsit_G, R_G, dR_G in zip(
                        lam_x, psitb.array,
                        dpsit.array, Rb.array,
                        dR.array):
                    axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
                    axpy(lam, dR_G, R_G)  # R_G += lam** dR_G

            # Final trial step
            with self.timer('precondition'):
                self.preconditioner(Rb.array, kpt, ekin_x, out=dpsit.array)

            self.timer.start('Update psi')
            if self.trial_step is not None:
                lam_x[:] = self.trial_step
            for lam, psit_G, dpsit_G in zip(lam_x, psitb.array, dpsit.array):
                axpy(lam, dpsit_G, psit_G)  # psit_G += lam * dpsit_G
            self.timer.stop('Update psi')

        self.timer.stop('RMM-DIIS')
        return error

    def __repr__(self):
        repr_string = 'RMM-DIIS eigensolver\n'
        repr_string += '       keep_htpsit: %s\n' % self.keep_htpsit
        repr_string += '       DIIS iterations: %d\n' % self.niter
        repr_string += '       Threshold for DIIS: %5.1e\n' % self.rtol
        repr_string += '       Limit lambda: %s\n' % self.limit_lambda
        repr_string += '       use_rayleigh: %s\n' % self.use_rayleigh
        repr_string += '       trial_step: %s' % self.trial_step
        return repr_string
