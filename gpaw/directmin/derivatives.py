import numpy as np
from gpaw.directmin.etdm import random_a, get_n_occ
from ase.units import Hartree
from gpaw.mpi import world
from gpaw.io.logger import GPAWLogger
from copy import deepcopy


class Derivatives:

    def __init__(self, etdm, wfs, c_ref=None, a_vec_u=None,
                 update_c_ref=False, eps=1.0e-7, random_amat=False):
        """
        :param etdm:
        :param wfs:
        :param c_ref: reference orbitals C_ref
        :param a_vec_u: skew-Hermitian matrix A
        :param update_c_ref: if True update reference orbitals
        :param eps: finite difference displacement
        :param random_amat: if True, use random matrix A
        """

        self.eps = eps

        # initialize vectors of elements matrix A
        if a_vec_u is None:
            self.a_vec_u = {u: np.zeros_like(v)
                            for u, v in etdm.a_vec_u.items()}

        if random_amat:
            for kpt in wfs.kpt_u:
                u = etdm.kpointval(kpt)
                a = random_a(etdm.a_vec_u[u].shape, wfs.dtype)
                wfs.gd.comm.broadcast(a, 0)
                self.a_vec_u[u] = a

        # initialize orbitals:
        if c_ref is None:
            self.c_ref = etdm.dm_helper.reference_orbitals
        else:
            self.c_ref = c_ref

        # update ref orbitals if needed
        if update_c_ref:
            etdm.rotate_wavefunctions(wfs, self.a_vec_u, etdm.n_dim,
                                      self.c_ref)
            etdm.dm_helper.set_reference_orbitals(wfs, etdm.n_dim)
            self.c_ref = etdm.dm_helper.reference_orbitals
            self.a_vec_u = {u: np.zeros_like(v)
                            for u, v in etdm.a_vec_u.items()}

    def get_analytical_derivatives(self, etdm, ham, wfs, dens,
                                   what2calc='gradient'):
        """
           Calculate analytical gradient or approximation to the Hessian
           with respect to the elements of a skew-Hermitian matrix

        :param etdm:
        :param ham:
        :param wfs:
        :param dens:
        :param what2calc: calculate gradient or Hessian
        :return: analytical gradient or Hessian
        """

        assert what2calc in ['gradient', 'hessian']

        if what2calc == 'gradient':
            # calculate analytical gradient
            analytical_der = etdm.get_energy_and_gradients(self.a_vec_u,
                                                           etdm.n_dim,
                                                           ham, wfs, dens,
                                                           self.c_ref)[1]
        else:
            # Calculate analytical approximation to hessian
            analytical_der = np.hstack([etdm.get_hessian(kpt).copy()
                                        for kpt in wfs.kpt_u])
            analytical_der = construct_real_hessian(analytical_der)
            analytical_der = np.diag(analytical_der)

        return analytical_der

    def get_numerical_derivatives(self, etdm, ham, wfs, dens,
                                  what2calc='gradient'):
        """
           Calculate numerical gradient or Hessian with respect to
           the elements of a skew-Hermitian matrix using central finite
           differences

        :param etdm:
        :param ham:
        :param wfs:
        :param dens:
        :param what2calc: calculate gradient or Hessian
        :return: numerical gradient or Hessian
        """

        assert what2calc in ['gradient', 'hessian']

        # total dimensionality if matrices are real
        dim = sum([len(a) for a in self.a_vec_u.values()])
        steps = [1.0, 1.0j] if etdm.dtype == complex else [1.0]
        use_energy_or_gradient = {'gradient': 0, 'hessian': 1}

        matrix_exp = etdm.matrix_exp
        if what2calc == 'gradient':
            numerical_der = {u: np.zeros_like(v)
                             for u, v in self.a_vec_u.items()}
        else:
            numerical_der = np.zeros(shape=(len(steps) * dim,
                                            len(steps) * dim))
            # have to use exact gradient when Hessian is calculated
            etdm.matrix_exp = 'egdecomp'

        row = 0
        f = use_energy_or_gradient[what2calc]
        for step in steps:
            for kpt in wfs.kpt_u:
                u = etdm.kpointval(kpt)
                for i in range(len(self.a_vec_u[u])):
                    a = self.a_vec_u[u][i]

                    self.a_vec_u[u][i] = a + step * self.eps
                    fplus = etdm.get_energy_and_gradients(
                        self.a_vec_u, etdm.n_dim, ham, wfs, dens,
                        self.c_ref)[f]

                    self.a_vec_u[u][i] = a - step * self.eps
                    fminus = etdm.get_energy_and_gradients(
                        self.a_vec_u, etdm.n_dim, ham, wfs, dens,
                        self.c_ref)[f]

                    derf = apply_central_finite_difference_approx(
                        fplus, fminus, self.eps)

                    if what2calc == 'gradient':
                        numerical_der[u][i] += step * derf
                    else:
                        numerical_der[row] = construct_real_hessian(derf)

                    row += 1
                    self.a_vec_u[u][i] = a

        if what2calc == 'hessian':
            etdm.matrix_exp = matrix_exp

        return numerical_der


class Davidson:
    """
    Finite difference generalized Davidson partial diagonalizer to obtain a
    number of the eigenpairs with the smallest eigenvalues.

    The following array indexation convention is used:

    e: Target eigenpair
    w: Krylov subspace
    """

    def __init__(self, etdm, logfile, fd_mode=None, m=None, h=None,
                 eps=None, cap_krylov=None, gmf=False,
                 accurate_first_pdiag=True, remember_sp_order=False,
                 sp_order=None, seed=None):
        """
        :param etdm: ETDM object for which the partial eigendecomposition
                     should be performed.
        :param logfile: Name string of the Davidson log file. Use '-' for
                        stdout or None to discard.
        :param fd_mode: Finite difference mode for partial Hessian evaluation.
                        Must be one of 'central' for central FD or 'forward'
                        for forward FD. Central FD uses two e/g evaluations per
                        Davidson step and target eigenpair with an error
                        scaling as O(h^2), forward FD uses one with O(h)
                        scaling.
        :param m: Memory parameter indicating how large the Krylov space should
                  be able to become before resetting it with the Ritz vectors
                  of the previous step or terminating the calculation if
                  cap_krylov is True.
        :param h: Displacement (in radians of orbital rotation) for finite
                  difference partial Hessian calculation.
        :param eps: Convergence threshold for maximum component of the
                    residuals of the target eigenpairs.
        :param cap_krylov: If True, terminate the calculation if the Krylov
                           space contains more than m vectors.
        :param gmf: Toggle usage with generalized mode following instead of
                    stability analysis. The defaults and some actions will be
                    different.
        :param accurate_first_pdiag: Approximate the target saddle point order
                                     better by performing a more accurate first
                                     partial diagonalization step at the
                                     expense of more gradient evaluations
        :param remember_sp_order: If True the number of target eigenpairs is
                                  saved after converging the partial Hessian
                                  eigendecomposition once and recovered for all
                                  subsequent calculations. If False the number
                                  of target eigenpairs is always gathered from
                                  the diagonal Hessian approximation in ETDM.
        :param sp_order: If given use this value for the number of target
                         eigenpairs instead of gathering it from the diagonal
                         Hessian approximation in ETDM.
        :param seed: Seed for random perturbation of initial Krylov space.
        """

        self.gmf = gmf
        self.etdm = etdm
        self.fd_mode = fd_mode
        self.remember_sp_order = remember_sp_order
        self.sp_order = sp_order
        self.log_sp_order_once = True
        self.seed = seed
        self.V_w = []       # Krylov subspace
        self.C_we = []      # Preconditioner
        self.M = []         # Matrix used to get preconditioner every step
        self.W_w = []       # Hessian effect on Krylov subspace
        # Rayleigh matrix, smaller representation of the
        # diagonalization problem to solve:
        self.H_ww = []
        self.lambda_e = []  # Target eigenvalues
        self.y_e = []       # Target eigenvectors in subspace representation
        self.x_e = []       # Target eigenvectors
        self.r_e = []       # Residuals of target eigenvectors
        self.t_e = []       # Krylov space extension vectors
        self.l = None       # Number of target eigenpairs
        self.h = h
        self.m = m
        self.converged_e = False
        self.all_converged = False
        self.error_e = []
        self.n_iter = 0
        self.eigenvalues = []
        self.eigenvectors = []
        self.reset = False
        self.eps = eps
        self.grad = None
        self.cap_krylov = cap_krylov
        self.dim_u = {}
        self.dimtot = 0
        self.nocc = {}
        self.nbands = 0
        self.c_ref = []
        self.logfile = logfile
        self.logger = GPAWLogger(world)
        self.logger.fd = logfile
        self.first_run = accurate_first_pdiag
        self.lambda_w = []  # All eigenvalues
        self.y_w = []       # All eigenvectors in subspace representation
        self.x_w = []       # All eigenvectors
        self.check_inputs()

    def check_inputs(self):
        defaults = self.set_defaults()
        assert self.etdm.name == 'etdm', 'Check etdm.'
        if self.logfile is not None:
            assert isinstance(self.logfile, str), 'Check logfile.'
        if self.fd_mode is None:
            self.fd_mode = defaults['fd_mode']
        else:
            assert self.fd_mode in ['central', 'forward'], 'Check fd_mode.'
        if self.m is None:
            self.m = defaults['m']
        else:
            assert isinstance(self.m, int) or np.isinf(self.m), 'Check m.'
        if self.h is None:
            self.h = defaults['h']
        else:
            assert isinstance(self.h, float), 'Check h.'
        if self.eps is None:
            self.eps = defaults['eps']
        else:
            assert isinstance(self.eps, float), 'Check eps.'
        if self.cap_krylov is None:
            self.cap_krylov = defaults['cap_krylov']
        else:
            assert isinstance(self.cap_krylov, bool), 'Check cap_krylov.'
        if self.remember_sp_order is None:
            self.remember_sp_order = defaults['remember_sp_order']
        else:
            assert isinstance(self.remember_sp_order, bool), \
                'Check remember_sp_order.'
        if self.sp_order is not None:
            assert isinstance(self.sp_order, int), 'Check sp_order.'

    def set_defaults(self):
        if self.gmf:
            return {'fd_mode': 'forward',
                    'm': 300,
                    'h': 1e-3,
                    'eps': 1e-2,
                    'cap_krylov': True,
                    'remember_sp_order': True}
        else:
            return {'fd_mode': 'central',
                    'm': np.inf,
                    'h': 1e-3,
                    'eps': 1e-3,
                    'cap_krylov': False,
                    'remember_sp_order': False}

    def todict(self):
        return {'name': 'Davidson',
                'logfile': self.logfile,
                'fd_mode': self.fd_mode,
                'm': self.m,
                'h': self.h,
                'eps': self.eps,
                'cap_krylov': self.cap_krylov,
                'gmf': self.gmf,
                'remember_sp_order': self.remember_sp_order,
                'sp_order': self.sp_order}

    def introduce(self):
        self.logger(
            '|-------------------------------------------------------|')
        self.logger(
            '|             Davidson partial diagonalizer             |')
        self.logger(
            '|-------------------------------------------------------|\n',
            flush=True)

    def run(self, wfs, ham, dens, use_prev=False):
        self.initialize(wfs, use_prev)
        if not self.gmf:
            self.etdm.sort_orbitals_mom(wfs)
        self.n_iter = 0
        self.c_ref = [deepcopy(wfs.kpt_u[x].C_nM)
                      for x in range(len(wfs.kpt_u))]
        if self.fd_mode == 'forward' and self.grad is None:
            self.obtain_grad_at_c_ref(wfs, ham, dens)
        while not self.all_converged:
            self.iterate(wfs, ham, dens)
        if self.remember_sp_order:
            if self.sp_order is None:
                self.determine_sp_order_from_lambda()
                self.logger(
                    'Saved target saddle point order as '
                    + str(self.sp_order) + ' for future partial '
                    'diagonalizations.', flush=True)
            elif self.log_sp_order_once:
                self.log_sp_order_once = False
                self.logger(
                    'Using target saddle point order of '
                    + str(self.sp_order) + '.', flush=True)
        if self.gmf:
            self.obtain_x_in_krylov_subspace()
        for k, kpt in enumerate(wfs.kpt_u):
            kpt.C_nM = deepcopy(self.c_ref[k])
        if not self.gmf:
            for kpt in wfs.kpt_u:
                self.etdm.sort_orbitals(ham, wfs, kpt)
        self.first_run = False

    def obtain_grad_at_c_ref(self, wfs, ham, dens):
        a_vec_u = {}
        n_dim = {}
        for k, kpt in enumerate(wfs.kpt_u):
            n_dim[k] = wfs.bd.nbands
            a_vec_u[k] = np.zeros_like(self.etdm.a_vec_u[k])
        self.grad = self.etdm.get_energy_and_gradients(
            a_vec_u, n_dim, ham, wfs, dens, self.c_ref)[1]

    def determine_sp_order_from_lambda(self):
        sp_order = 0
        for i in range(len(self.lambda_w)):
            if self.lambda_w[i] < 1e-8:
                sp_order += 1
            else:
                break
        self.sp_order = sp_order
        if self.sp_order == 0:
            self.sp_order = 1

    def obtain_x_in_krylov_subspace(self):
        self.x_w = []
        for i in range(len(self.lambda_w)):
            self.x_w.append(
                self.V_w[:, :len(self.lambda_w)] @ self.y_w[i].T)
        self.x_w = np.asarray(self.x_w).T

    def initialize(self, wfs, use_prev=False):
        """
        This is separate from __init__ since the initial Krylov space is
        obtained here every time a partial diagonalization is performed at
        different electronic coordinates.
        """

        dimz = 2 if self.etdm.dtype == complex else 1
        self.introduce()
        self.reset = False
        self.all_converged = False
        self.l = 0
        self.V_w = None
        self.nbands = wfs.bd.nbands
        appr_hess, appr_sp_order = self.estimate_spo_and_update_appr_hess(
            wfs, use_prev=use_prev)
        self.M = np.zeros(shape=self.dimtot * dimz)
        for i in range(self.dimtot * dimz):
            self.M[i] = np.real(appr_hess[i % self.dimtot])
        if self.sp_order is not None:
            self.l = self.sp_order
        else:
            self.l = appr_sp_order if self.gmf else appr_sp_order + 2
        if self.l == 0:
            self.l = 1
        if self.l > self.dimtot * dimz:
            self.l = self.dimtot * dimz
        self.W_w = None
        self.error_e = [np.inf for x in range(self.l)]
        self.converged_e = [False for x in range(self.l)]
        self.form_initial_krylov_subspace(
            wfs, appr_hess, dimz, use_prev=use_prev)
        text = 'Davidson will target the ' + str(self.l) + ' lowest eigenpairs'
        if self.sp_order is None:
            text += '.'
        else:
            text += ' as recovered from previous calculation.'
        self.logger(text, flush=True)

    def get_approximate_hessian_and_dim(self, wfs):
        appr_hess = []
        self.dimtot = 0
        for k, kpt in enumerate(wfs.kpt_u):
            hdia = self.etdm.get_hessian(kpt)
            self.dim_u[k] = len(hdia)
            self.dimtot += len(hdia)
            appr_hess += list(hdia.copy())
            self.nocc[k] = get_n_occ(kpt)
        return appr_hess

    def estimate_spo_and_update_appr_hess(self, wfs, use_prev=False):
        appr_sp_order = 0
        appr_hess = self.get_approximate_hessian_and_dim(wfs)
        if use_prev:
            for i in range(len(self.lambda_w)):
                if self.lambda_w[i] < -1e-4:
                    appr_sp_order += 1
                    if self.etdm.dtype == complex:
                        appr_hess[i] \
                            = self.lambda_w[i] + 1.0j * self.lambda_w[i]
                    else:
                        appr_hess[i] = self.lambda_w[i]
        else:
            for i in range(len(appr_hess)):
                if np.real(appr_hess[i]) < -1e-4:
                    appr_sp_order += 1
        return appr_hess, appr_sp_order

    def form_initial_krylov_subspace(
            self, wfs, appr_hess, dimz, use_prev=False):
        rng = np.random.default_rng(self.seed)
        wfs.timer.start('Initial Krylov subspace')
        if use_prev:
            self.randomize_krylov_subspace(rng, dimz)
        else:
            self.initialize_randomized_krylov_subspace(rng, dimz, appr_hess)
        wfs.timer.start('Modified Gram-Schmidt')
        self.V_w = mgs(self.V_w)
        wfs.timer.stop('Modified Gram-Schmidt')
        self.V_w = self.V_w.T
        wfs.timer.stop('Initial Krylov subspace')

    def randomize_krylov_subspace(self, rng, dimz, reps=1e-4):
        self.V_w = deepcopy(self.x_w.T[: self.l])
        for i in range(self.l):
            for k in range(self.dimtot):
                for l in range(dimz):
                    rand = np.zeros(shape=2)
                    if world.rank == 0:
                        rand[0] = rng.random()
                        rand[1] = 1 if rng.random() > 0.5 else -1
                    else:
                        rand[0] = 0.0
                        rand[1] = 0.0
                    world.broadcast(rand, 0)
                    self.V_w[i][l * self.dimtot + k] \
                        += rand[1] * reps * rand[0]

    def initialize_randomized_krylov_subspace(
            self, rng, dimz, appr_hess, reps=1e-4):
        do_conj = False

        # Just for F821
        v = None
        imin = None

        self.V_w = []
        for i in range(self.l):
            if do_conj:
                v[self.dimtot + imin] = -1.0
                do_conj = False
            else:
                v = np.zeros(self.dimtot * dimz)
                rdia = np.real(appr_hess).copy()
                imin = int(np.where(rdia == min(rdia))[0][0])
                rdia[imin] = np.inf
                v[imin] = 1.0
                if self.etdm.dtype == complex:
                    v[self.dimtot + imin] = 1.0
                    do_conj = True
            for l in range(self.dimtot):
                for m in range(dimz):
                    if l == imin:
                        continue
                    rand = np.zeros(shape=2)
                    if world.rank == 0:
                        rand[0] = rng.random()
                        rand[1] = 1 if rng.random() > 0.5 else -1
                    else:
                        rand[0] = 0.0
                        rand[1] = 0.0
                    world.broadcast(rand, 0)
                    v[m * self.dimtot + l] = rand[1] * reps * rand[0]
            self.V_w.append(v / np.linalg.norm(v))
        self.V_w = np.asarray(self.V_w)

    def iterate(self, wfs, ham, dens):
        self.t_e = []
        self.evaluate_W(wfs, ham, dens)
        wfs.timer.start('Rayleigh matrix formation')
        self.H_ww = self.V_w.T @ self.W_w
        wfs.timer.stop('Rayleigh matrix formation')
        self.n_iter += 1
        wfs.timer.start('Rayleigh matrix diagonalization')
        eigv, eigvec = np.linalg.eigh(self.H_ww)
        wfs.timer.stop('Rayleigh matrix diagonalization')
        eigvec = eigvec.T
        self.lambda_w = deepcopy(eigv)
        self.y_w = deepcopy(eigvec)
        self.lambda_e = eigv[: self.l]
        self.y_e = eigvec[: self.l]
        self.calculate_ritz_vectors(wfs)
        self.calculate_residuals(wfs)
        for i in range(self.l):
            self.error_e[i] = np.abs(self.r_e[i]).max()
        self.all_converged = True
        for i in range(self.l):
            self.converged_e[i] = self.error_e[i] < self.eps
            self.all_converged = self.all_converged and self.converged_e[i]
        if self.all_converged:
            self.eigenvalues = deepcopy(self.lambda_e)
            self.eigenvectors = deepcopy(self.x_e)
            self.log()
            return
        self.calculate_preconditioner(wfs)
        self.augment_krylov_subspace(wfs)
        self.log()

    def evaluate_W(self, wfs, ham, dens):
        wfs.timer.start('FD Hessian vector product')
        if self.W_w is None:
            self.W_w = []
            Vt = self.V_w.T
            for i in range(len(Vt)):
                self.W_w.append(self.get_fd_hessian(Vt[i], wfs, ham, dens))
            self.reset = False
        else:
            added = len(self.V_w[0]) - len(self.W_w[0])
            self.W_w = self.W_w.T.tolist()
            Vt = self.V_w.T
            for i in range(added):
                self.W_w.append(self.get_fd_hessian(
                    Vt[-added + i], wfs, ham, dens))
        self.W_w = np.asarray(self.W_w).T
        wfs.timer.stop('FD Hessian vector product')

    def calculate_ritz_vectors(self, wfs):
        wfs.timer.start('Ritz vector calculation')
        self.x_e = []
        for i in range(self.l):
            self.x_e.append(self.V_w @ self.y_e[i].T)
        self.x_e = np.asarray(self.x_e)
        wfs.timer.stop('Ritz vector calculation')

    def calculate_residuals(self, wfs):
        wfs.timer.start('Residual calculation')
        self.r_e = []
        for i in range(self.l):
            self.r_e.append(
                self.x_e[i] * self.lambda_e[i] - self.W_w @ self.y_e[i].T)
        self.r_e = np.asarray(self.r_e)
        wfs.timer.stop('Residual calculation')

    def calculate_preconditioner(self, wfs):
        n_dim = len(self.V_w)
        wfs.timer.start('Preconditioner calculation')
        self.C_we = np.zeros(shape=(self.l, n_dim))
        for i in range(self.l):
            self.C_we[i] = -np.abs(
                np.repeat(self.lambda_e[i], n_dim) - self.M) ** -1
            for l in range(len(self.C_we[i])):
                if self.C_we[i][l] > -0.1 * Hartree:
                    self.C_we[i][l] = -0.1 * Hartree
        wfs.timer.stop('Preconditioner calculation')

    def augment_krylov_subspace(self, wfs):
        wfs.timer.start('Krylov subspace augmentation')
        self.get_new_krylov_subspace_directions(wfs)
        self.V_w = np.asarray(self.V_w)
        if self.cap_krylov:
            if len(self.V_w) > self.m:
                self.logger(
                    'Krylov space exceeded maximum size. Partial '
                    'diagonalization is not fully converged. Current size '
                    'is ' + str(len(self.V_w) - len(self.t_e)) + '. Size at '
                    'next step would be ' + str(len(self.V_w)) + '.',
                    flush=True)
                self.all_converged = True
        wfs.timer.start('Modified Gram-Schmidt')
        self.V_w = mgs(self.V_w)
        wfs.timer.stop('Modified Gram-Schmidt')
        self.V_w = self.V_w.T
        wfs.timer.stop('Krylov subspace augmentation')

    def get_new_krylov_subspace_directions(self, wfs):
        wfs.timer.start('New directions')
        for i in range(self.l):
            if not self.converged_e[i] or (self.gmf and self.first_run):
                self.t_e.append(self.C_we[i] * self.r_e[i])
        self.t_e = np.asarray(self.t_e)
        if len(self.V_w[0]) <= self.m:
            self.V_w = self.V_w.T.tolist()
            for i in range(len(self.t_e)):
                self.V_w.append(self.t_e[i])
        elif not self.cap_krylov:
            self.reset = True
            self.V_w = deepcopy(self.x_e.tolist())
            for i in range(len(self.t_e)):
                self.V_w.append(self.t_e[i])
            self.W_w = None
        wfs.timer.stop('New directions')

    def log(self):
        self.logger('Dimensionality of Krylov space: '
                    + str(len(self.V_w[0]) - len(self.t_e)), flush=True)
        if self.reset:
            self.logger('Reset Krylov space', flush=True)
        self.logger('\nEigenvalues:\n', flush=True)
        text = ''
        for i in range(self.l):
            text += '%10d '
        indices = text % tuple(range(1, self.l + 1))
        self.logger(indices, flush=True)
        text = ''
        for i in range(self.l):
            text += '%10.6f '
        self.logger(text % tuple(self.lambda_e), flush=True)
        self.logger('\nResidual maximum components:\n', flush=True)
        self.logger(indices, flush=True)
        text = ''
        temp = list(np.round(deepcopy(self.error_e), 6))
        for i in range(self.l):
            text += '%10s '
            temp[i] = str(temp[i])
            if self.converged_e[i]:
                temp[i] += 'c'
        self.logger(text % tuple(temp), flush=True)

    def get_fd_hessian(self, vin, wfs, ham, dens):
        """
        Get the dot product of the Hessian and a vector with a finite
        difference approximation.

        :param vin: The vector
        :param wfs:
        :param ham:
        :param dens:
        :return: Dot product vector of the Hessian and the vector.
        """

        v = self.h * vin
        c_ref = deepcopy(self.c_ref)
        gp = self.calculate_displaced_grad(wfs, ham, dens, c_ref, v)
        hessi = []
        if self.fd_mode == 'central':
            gm = self.calculate_displaced_grad(wfs, ham, dens, c_ref, -1.0 * v)
            for k in range(len(wfs.kpt_u)):
                hessi += list((gp[k] - gm[k]) * 0.5 / self.h)
        elif self.fd_mode == 'forward':
            for k in range(len(wfs.kpt_u)):
                hessi += list((gp[k] - self.grad[k]) / self.h)
        for k, kpt in enumerate(wfs.kpt_u):
            kpt.C_nM = c_ref[k]  # LCAO-specific
        dens.update(wfs)
        if self.etdm.dtype == complex:
            hessc = np.zeros(shape=(2 * self.dimtot))
            hessc[: self.dimtot] = np.real(hessi)
            hessc[self.dimtot:] = np.imag(hessi)
            return hessc
        else:
            return np.asarray(hessi)

    def calculate_displaced_grad(self, wfs, ham, dens, c_ref, v):
        a_vec_u = {}
        n_dim = {}
        start = 0
        end = 0
        for k in range(len(wfs.kpt_u)):
            a_vec_u[k] = np.zeros_like(self.etdm.a_vec_u[k])
            n_dim[k] = wfs.bd.nbands
            end += self.dim_u[k]
            a_vec_u[k] += v[start: end]
            if self.etdm.dtype == complex:
                a_vec_u[k] += 1.0j * v[self.dimtot + start: self.dimtot + end]
            start += self.dim_u[k]
        return self.etdm.get_energy_and_gradients(
            a_vec_u, n_dim, ham, wfs, dens, c_ref)[1]

    def break_instability(self, wfs, n_dim, c_ref, number,
                          initial_guess='displace', ham=None, dens=None):
        """
        Displaces orbital rotation coordinates in the direction of an
        instability. Uses a fixed displacement or performs a line search.

        :param wfs:
        :param n_dim:
        :param c_ref:
        :param number: Instability index
        :param initial_guess: How to displace. Can be one of the following:
        displace: Use a fixed displacement; line_search: Performs a
        backtracking line search.
        :param ham:
        :param dens:
        """

        assert self.converged_e, 'Davidson cannot break instabilities since' \
            + ' the partial eigendecomposition has not been converged.'
        assert len(self.lambda_e) >= number, 'Davidson cannot break' \
            + ' instability no. ' + str(number) + ' since this eigenpair was' \
            + 'not converged.'
        assert self.lambda_e[number - 1] < 0.0, 'Eigenvector no. ' \
            + str(number) + ' does not represent an instability.'

        step = self.etdm.line_search.max_step
        instability = step * self.x_e[number - 1]
        if initial_guess == 'displace':
            a_vec_u = self.displace(instability)
        elif initial_guess == 'line_search':
            assert ham is not None and dens is not None, 'Hamiltonian and' \
                                                         'density needed for' \
                                                         'line search.'
            a_vec_u = self.do_line_search(
                wfs, dens, ham, n_dim, c_ref, instability)
        self.etdm.rotate_wavefunctions(wfs, a_vec_u, n_dim, c_ref)

    def displace(self, instability):
        a_vec_u = deepcopy(self.etdm.a_vec_u)
        start = 0
        stop = 0
        for k in a_vec_u.keys():
            stop += self.dim_u[k]
            a_vec_u[k] = instability[start: stop]
            start += self.dim_u[k]
        return a_vec_u

    def do_line_search(self, wfs, dens, ham, n_dim, c_ref, instability):
        a_vec_u = {}
        p_vec_u = {}
        start = 0
        stop = 0
        for k in a_vec_u.keys():
            stop += self.dim_u[k]
            p_vec_u[k] = instability[start: stop]
            start += self.dim_u[k]
        phi, g_vec_u = self.etdm.get_energy_and_gradients(
            a_vec_u, n_dim, ham, wfs, dens, c_ref)
        der_phi = 0.0
        for k in g_vec_u:
            der_phi += g_vec_u[k].conj() @ p_vec_u[k]
        der_phi = der_phi.real
        der_phi = wfs.kd.comm.sum(der_phi)
        alpha = self.etdm.line_search.step_length_update(
            a_vec_u, p_vec_u, n_dim, ham, wfs, dens, c_ref,
            phi_0=phi, der_phi_0=der_phi, phi_old=None,
            der_phi_old=None, alpha_max=5.0, alpha_old=None,
            kpdescr=wfs.kd)[0]
        for k in a_vec_u.keys():
            a_vec_u[k] = alpha * p_vec_u[k]
        return a_vec_u

    def estimate_sp_order(self, calc, method='appr-hess', target_more=1):
        self.etdm.sort_orbitals_mom(calc.wfs)
        constraints_copy = deepcopy(self.etdm.constraints)
        self.etdm.constraints = [[] for _ in range(len(calc.wfs.kpt_u))]
        appr_hess, appr_sp_order = self.estimate_spo_and_update_appr_hess(
            calc.wfs)
        if method == 'full-hess':
            self.sp_order = appr_sp_order + target_more
            self.run(calc.wfs, calc.hamiltonian, calc.density)
            appr_hess, appr_sp_order = self.estimate_spo_and_update_appr_hess(
                calc.wfs, use_prev=True)
        for kpt in calc.wfs.kpt_u:
            self.etdm.sort_orbitals(calc.hamiltonian, calc.wfs, kpt)
        self.etdm.constraints = deepcopy(constraints_copy)
        return appr_sp_order


def mgs(vin):
    """
    Modified Gram-Schmidt orthonormalization

    :param vin: Set of vectors.
    :return: Orthonormal set of vectors.
    """

    v = deepcopy(vin)
    q = np.zeros_like(v)
    for i in range(len(v)):
        q[i] = v[i] / np.linalg.norm(v[i])
        for k in range(len(v)):
            v[k] = v[k] - np.dot(np.dot(q[i].T, v[k]), q[i])
    return q


def construct_real_hessian(hess):

    if hess.dtype == complex:
        hess_real = np.hstack((np.real(hess), np.imag(hess)))
    else:
        hess_real = hess

    return hess_real


def apply_central_finite_difference_approx(fplus, fminus, eps):

    if isinstance(fplus, dict) and isinstance(fminus, dict):
        assert (len(fplus) == len(fminus))
        derf = np.hstack([(fplus[k] - fminus[k]) * 0.5 / eps
                          for k in fplus.keys()])
    elif isinstance(fplus, float) and isinstance(fminus, float):
        derf = (fplus - fminus) * 0.5 / eps
    else:
        raise ValueError()

    return derf
