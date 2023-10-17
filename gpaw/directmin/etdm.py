"""
A class for finding optimal
orbitals of the KS-DFT or PZ-SIC
functionals using exponential transformation
direct minimization

arXiv:2101.12597 [physics.comp-ph]
Comput. Phys. Commun. 267, 108047 (2021).
https://doi.org/10.1016/j.cpc.2021.108047
"""


import numpy as np
import warnings
from gpaw.directmin.tools import expm_ed, expm_ed_unit_inv
from gpaw.directmin.lcao.directmin_lcao import DirectMinLCAO
from scipy.linalg import expm
from gpaw.directmin import search_direction, line_search_algorithm
from gpaw.directmin.functional import get_functional
from gpaw import BadParallelization
from copy import deepcopy


class ETDM:

    """
    Exponential Transformation Direct Minimization (ETDM)
    """

    def __init__(self,
                 searchdir_algo='l-bfgs-p',
                 linesearch_algo='swc-awc',
                 partial_diagonalizer='Davidson',
                 update_ref_orbs_counter=20,
                 update_ref_orbs_canonical=False,
                 update_precond_counter=1000,
                 use_prec=True, matrix_exp='pade-approx',
                 representation='sparse',
                 functional='ks',
                 orthonormalization='gramschmidt',
                 randomizeorbitals=False,
                 checkgraderror=False,
                 localizationtype=None,
                 need_localization=True,
                 need_init_orbs=True,
                 constraints=None,
                 ):
        """
        This class performs the exponential transformation
        direct minimization:
        E = E[C_ref e^{A}]
        C_ref are the reference orbitals
        A is a skew-Hermitian matrix
        We want to find A that optimizes the orbitals. The optimal orbitals
        are those corresponding to a minimum for the ground state or a saddle
        point for excited states.

        :param searchdir_algo: algorithm for calculating the search direction
        (e.g.LBFGS)
        :param linesearch_algo: line search (e.g. strong Wolfe conditions)
        :param partial_diagonalizer: Algorithm to use for partial
        diagonalization of the electronic Hessian if DO-GMF is used
        :param update_ref_orbs_counter: When to update C_ref
        :param update_ref_orbs_canonical: update C_ref to canonical orbitals
        :param update_precond_counter: when to update the preconditioner
        :param use_prec: use preconditioner or not
        :param matrix_exp: algorithm for calculating the matrix exponential and
        gradient. Can be one of 'pade-approx', 'egdecomp', 'egdecomp-u-invar'
        (used with u-invar representation)
        :param representation: the way the elements of A are stored. Can be one
        of 'sparse', 'full', 'u-invar'
        :param functional: KS or PZ-SIC
        :param orthonormalization: orthonormalize the orbitals using
        Gram-Schmidt or Loewdin orthogonalization, or getting the orbitals as
        eigenstates of the Hamiltonian matrix. Can be one of 'gramschmidt',
        'loewdin', 'diag'
        :param randomizeorbitals: if True, add noise to the initial guess
        :param checkgraderror: check error in estimation of gradient
        :param localizationtype: Foster-Boys, Pipek-Mezey, Edm.-Rudenb.
        :param need_localization: use localized orbitals as initial guess
        :param need_init_orbs: if false, then use orbitals stored in kpt
        :param constraints: List of constraints for each kpt. Can be given
        either as pairs of orbital indices or single indices which are
        converted to list of all pairs involving the single index
        """

        assert representation in ['sparse', 'u-invar', 'full'], 'Value Error'
        assert matrix_exp in ['egdecomp', 'egdecomp-u-invar', 'pade-approx'], \
            'Value Error'
        if matrix_exp == 'egdecomp-u-invar':
            assert representation == 'u-invar', 'Use u-invar representation ' \
                                                'with egdecomp-u-invar'
        assert orthonormalization in ['gramschmidt', 'loewdin', 'diag'], \
            'Value Error'

        self.localizationtype = localizationtype
        self.eg_count = 0
        self.update_ref_orbs_counter = update_ref_orbs_counter
        self.update_ref_orbs_canonical = update_ref_orbs_canonical
        self.update_precond_counter = update_precond_counter
        self.use_prec = use_prec
        self.matrix_exp = matrix_exp
        self.iters = 0
        self.restart = False
        self.name = 'etdm'
        self.localizationtype = localizationtype
        self.need_localization = need_localization
        self.need_init_orbs = need_init_orbs
        self.randomizeorbitals = randomizeorbitals
        self.representation = representation
        self.orthonormalization = orthonormalization
        self.constraints = constraints

        self.gmf = False
        self.searchdir_algo = search_direction(
            searchdir_algo, self, partial_diagonalizer)
        sd_name = self.searchdir_algo.name.split('_')
        if sd_name[0] == 'l-bfgs-p' and not self.use_prec:
            raise ValueError('Use l-bfgs-p with use_prec=True')
        if len(sd_name) == 2:
            if sd_name[1] == 'gmf':
                self.searchdir_algo.name = sd_name[0]
                self.gmf = True
                self.g_vec_u_original = None
                self.pd = partial_diagonalizer

        self.line_search = line_search_algorithm(linesearch_algo,
                                                 self.evaluate_phi_and_der_phi,
                                                 self.searchdir_algo)
        self.func = get_functional(functional)

        self.checkgraderror = checkgraderror
        self._norm_commutator, self._norm_grad = 0., 0.
        self.error = 0

        # these are things we cannot initialize now
        self.dtype = None
        self.nkpts = None
        self.gd = None
        self.nbands = None

        # values: vectors of the elements of matrices, keys: kpt number
        self.a_vec_u = {}  # for the elements of the skew-Hermitian matrix A
        self.g_vec_u = {}  # for the elements of the gradient matrix G
        self.evecs = {}   # eigenvectors for i*a_vec_u
        self.evals = {}   # eigenvalues for i*a_vec_u
        self.ind_up = {}
        self.n_dim = {}
        self.alpha = 1.0  # step length
        self.phi_2i = [None, None]  # energy at last two iterations
        self.der_phi_2i = [None, None]  # energy gradient w.r.t. alpha
        self.hess = {}  # approximate Hessian

        # for mom
        self.initial_occupation_numbers = None

        # in this attribute we store the object specific to each mode
        self.dm_helper = None

        self.initialized = False

    def __repr__(self):

        sda_name = self.searchdir_algo.name
        lsa_name = self.line_search.name

        add = ''
        pd_add = ''
        if self.gmf:
            add = ' with minimum mode following'
            pardi = {'Davidson': 'Finite difference generalized Davidson '
                     'algorithm'}
            pd_add = '       ' \
                     'Partial diagonalizer: {}\n'.format(
                         pardi[self.pd['name']])

        sds = {'sd': 'Steepest Descent',
               'fr-cg': 'Fletcher-Reeves conj. grad. method',
               'quick-min': 'Molecular-dynamics based algorithm',
               'l-bfgs': 'L-BFGS algorithm',
               'l-bfgs-p': 'L-BFGS algorithm with preconditioning',
               'l-sr1p': 'Limited-memory SR1P algorithm'}

        lss = {'max-step': 'step size equals one',
               'parabola': 'Parabolic line search',
               'swc-awc': 'Inexact line search based on cubic interpolation,\n'
                          '                    strong and approximate Wolfe '
                          'conditions'}

        repr_string = 'Direct minimisation' + add + ' using exponential ' \
                      'transformation.\n'
        repr_string += '       ' \
                       'Search ' \
                       'direction: {}\n'.format(sds[sda_name] + add)
        repr_string += pd_add
        repr_string += '       ' \
                       'Line ' \
                       'search: {}\n'.format(lss[lsa_name])
        repr_string += '       ' \
                       'Preconditioning: {}\n'.format(self.use_prec)
        repr_string += '       ' \
                       'WARNING: do not use it for metals as ' \
                       'occupation numbers are\n' \
                       '                ' \
                       'not found variationally\n'

        return repr_string

    def initialize(self, gd, dtype, nbands, nkpts, nao, using_blacs,
                   bd_comm_size, kpt_u):

        assert nbands == nao, \
            'Please, use: nbands=\'nao\''
        if not bd_comm_size == 1:
            raise BadParallelization(
                'Band parallelization is not supported')
        if using_blacs:
            raise BadParallelization(
                'ScaLapack parallelization is not supported')

        self.gd = gd
        self.dtype = dtype
        self.nbands = nbands
        self.nkpts = nkpts

        for kpt in kpt_u:
            u = self.kpointval(kpt)
            # dimensionality of the problem.
            # this implementation rotates among all bands
            self.n_dim[u] = self.nbands

        self.initialized = True

        # need reference orbitals
        self.dm_helper = None

    def initialize_dm_helper(self, wfs, ham, dens):

        self.dm_helper = DirectMinLCAO(
            wfs, ham, self.nkpts,
            diagonalizer=None,
            orthonormalization=self.orthonormalization,
            need_init_orbs=self.need_init_orbs
        )
        self.need_init_orbs = self.dm_helper.need_init_orbs
        # mom
        wfs.calculate_occupation_numbers(dens.fixed)
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            self.initial_occupation_numbers = wfs.occupations.numbers.copy()
            self.initialize_mom(wfs, dens)

        for kpt in wfs.kpt_u:
            f_unique = np.unique(kpt.f_n)
            if len(f_unique) > 2 and self.representation == 'u-invar':
                warnings.warn("Use representation == 'sparse' when "
                              "there are unequally occupied orbitals "
                              "as the functional is not unitary invariant")

        # randomize orbitals?
        if self.randomizeorbitals:
            for kpt in wfs.kpt_u:
                self.randomize_orbitals_kpt(wfs, kpt)
            self.randomizeorbitals = False

        # initialize matrices
        self.set_variable_matrices(wfs.kpt_u)
        # if no empty state no need to optimize
        for k in self.ind_up:
            if not self.ind_up[k][0].size or not self.ind_up[k][1].size:
                self.n_dim[k] = 0
        # set reference orbitals
        self.dm_helper.set_reference_orbitals(wfs, self.n_dim)

    def set_variable_matrices(self, kpt_u):

        # Matrices are sparse and Skew-Hermitian.
        # They have this structure:
        #  A_BigMatrix =
        #
        # (  A_1          A_2 )
        # ( -A_2.T.conj() 0   )
        #
        # where 0 is a zero-matrix of size of (M-N) * (M-N)
        #
        # A_1 i skew-hermitian matrix of N * N,
        # N-number of occupied states
        # A_2 is matrix of size of (M-N) * N,
        # M - number of basis functions
        #
        # if the energy functional is unitary invariant
        # then A_1 = 0
        # (see Hutter J et. al, J. Chem. Phys. 101, 3862 (1994))
        #
        # We will keep A_1 as we would like to work with metals,
        # SIC, and molecules with different occupation numbers.
        # this corresponds to 'sparse' representation
        #
        # Thus, for the 'sparse' we need to store upper
        # triangular part of A_1, and matrix A_2, so in total
        # (M-N) * N + N * (N - 1)/2 = N * (M - (N + 1)/2) elements
        #
        # we will store these elements as a vector and
        # also will store indices of the A_BigMatrix
        # which correspond to these elements.
        #
        # 'u-invar' corresponds to the case when we want to
        # store only A_2, that is this representation is sparser

        for kpt in kpt_u:
            n_occ = get_n_occ(kpt)
            u = self.kpointval(kpt)
            # M - one dimension of the A_BigMatrix
            M = self.n_dim[u]
            if self.representation == 'u-invar':
                i1, i2 = [], []
                for i in range(n_occ):
                    for j in range(n_occ, M):
                        i1.append(i)
                        i2.append(j)
                self.ind_up[u] = (np.asarray(i1), np.asarray(i2))
            else:
                if self.representation == 'full' and self.dtype == complex:
                    # Take indices of all upper triangular and diagonal
                    # elements of A_BigMatrix
                    self.ind_up[u] = np.triu_indices(self.n_dim[u])
                else:
                    self.ind_up[u] = np.triu_indices(self.n_dim[u], 1)
                    if self.representation == 'sparse':
                        # Delete indices of elements that correspond
                        # to 0 matrix in A_BigMatrix
                        zero_ind = -((M - n_occ) * (M - n_occ - 1)) // 2
                        if zero_ind == 0:
                            zero_ind = None
                        self.ind_up[u] = (self.ind_up[u][0][:zero_ind].copy(),
                                          self.ind_up[u][1][:zero_ind].copy())

            shape_of_arr = len(self.ind_up[u][0])

            self.a_vec_u[u] = np.zeros(shape=shape_of_arr, dtype=self.dtype)
            self.g_vec_u[u] = np.zeros(shape=shape_of_arr, dtype=self.dtype)
            self.evecs[u] = None
            self.evals[u] = None

            # All constraints passed as a list of a single orbital index are
            # converted to all lists of orbital pairs involving that orbital
            # index
            if self.constraints:
                self.constraints[u] = convert_constraints(
                    self.constraints[u], self.n_dim[u],
                    len(kpt.f_n[kpt.f_n > 1e-10]), self.representation)

        # This conversion makes it so that constraint-related functions can
        # iterate through a list of no constraints rather than checking for
        # None every time
        if self.constraints is None:
            self.constraints = [[] for _ in range(len(kpt_u))]

        self.iters = 1

    def iterate(self, ham, wfs, dens):
        """
        One iteration of direct optimization
        for occupied orbitals

        :param ham:
        :param wfs:
        :param dens:
        :return:
        """
        with wfs.timer('Direct Minimisation step'):
            self.update_ref_orbitals(wfs, ham, dens)

            a_vec_u = self.a_vec_u
            n_dim = self.n_dim
            alpha = self.alpha
            phi_2i = self.phi_2i
            der_phi_2i = self.der_phi_2i
            c_ref = self.dm_helper.reference_orbitals

            if self.iters == 1:
                phi_2i[0], g_vec_u = \
                    self.get_energy_and_gradients(a_vec_u, n_dim, ham, wfs,
                                                  dens, c_ref)
            else:
                g_vec_u = self.g_vec_u_original if self.gmf else self.g_vec_u

            make_pd = False
            if self.gmf:
                with wfs.timer('Partial Hessian diagonalization'):
                    self.searchdir_algo.update_eigenpairs(
                        g_vec_u, wfs, ham, dens)
                # The diagonal Hessian approximation must be positive-definite
                make_pd = True

            with wfs.timer('Preconditioning:'):
                precond = self.get_preconditioning(
                    wfs, self.use_prec, make_pd=make_pd)

            with wfs.timer('Get Search Direction'):
                # calculate search direction according to chosen
                # optimization algorithm (e.g. L-BFGS)
                p_vec_u = self.searchdir_algo.update_data(wfs, a_vec_u,
                                                          g_vec_u, precond)

            # recalculate derivative with new search direction
            der_phi_2i[0] = 0.0
            for k in g_vec_u:
                der_phi_2i[0] += g_vec_u[k].conj() @ p_vec_u[k]
            der_phi_2i[0] = der_phi_2i[0].real
            der_phi_2i[0] = wfs.kd.comm.sum_scalar(der_phi_2i[0])

            alpha, phi_alpha, der_phi_alpha, g_vec_u = \
                self.line_search.step_length_update(a_vec_u, p_vec_u,
                                                    n_dim, ham, wfs, dens,
                                                    c_ref,
                                                    phi_0=phi_2i[0],
                                                    der_phi_0=der_phi_2i[0],
                                                    phi_old=phi_2i[1],
                                                    der_phi_old=der_phi_2i[1],
                                                    alpha_max=5.0,
                                                    alpha_old=alpha,
                                                    kpdescr=wfs.kd)

            if wfs.gd.comm.size > 1:
                with wfs.timer('Broadcast gradients'):
                    alpha_phi_der_phi = np.array([alpha, phi_2i[0],
                                                  der_phi_2i[0]])
                    wfs.gd.comm.broadcast(alpha_phi_der_phi, 0)
                    alpha = alpha_phi_der_phi[0]
                    phi_2i[0] = alpha_phi_der_phi[1]
                    der_phi_2i[0] = alpha_phi_der_phi[2]
                    for kpt in wfs.kpt_u:
                        k = self.kpointval(kpt)
                        wfs.gd.comm.broadcast(g_vec_u[k], 0)

            # calculate new matrices for optimal step length
            for k in a_vec_u:
                a_vec_u[k] += alpha * p_vec_u[k]
            self.alpha = alpha
            self.g_vec_u = g_vec_u
            self.iters += 1

            # and 'shift' phi, der_phi for the next iteration
            phi_2i[1], der_phi_2i[1] = phi_2i[0], der_phi_2i[0]
            phi_2i[0], der_phi_2i[0] = phi_alpha, der_phi_alpha,

    def get_energy_and_gradients(self, a_vec_u, n_dim, ham, wfs, dens,
                                 c_ref):

        """
        Energy E = E[C_ref exp(A)]. Gradients G_ij[C, A] = dE/dA_ij

        :param wfs:
        :param ham:
        :param dens:
        :param a_vec_u: A
        :param c_ref: C_ref
        :param n_dim:
        :return:
        """

        self.rotate_wavefunctions(wfs, a_vec_u, n_dim, c_ref)

        e_total = self.update_ks_energy(ham, wfs, dens)

        with wfs.timer('Calculate gradients'):
            g_vec_u = {}
            self.error = 0.0
            self.e_sic = 0.0  # this is odd energy
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                if n_dim[k] == 0:
                    g_vec_u[k] = np.zeros_like(a_vec_u[k])
                    continue
                g_vec_u[k], error = self.dm_helper.calc_grad(
                    wfs, ham, kpt, self.func, self.evecs[k], self.evals[k],
                    self.matrix_exp, self.representation, self.ind_up[k],
                    self.constraints[k])

                self.error += error
            self.error = wfs.kd.comm.sum(self.error)
            self.e_sic = wfs.kd.comm.sum(self.e_sic)

        self.eg_count += 1

        if self.representation == 'full' and self.checkgraderror:
            self._norm_commutator = 0.0
            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                a_mat = vec2skewmat(a_vec_u[u], self.n_dim[u],
                                    self.ind_up[u], wfs.dtype)
                g_mat = vec2skewmat(g_vec_u[u], self.n_dim[u],
                                    self.ind_up[u], wfs.dtype)

                tmp = np.linalg.norm(g_mat @ a_mat - a_mat @ g_mat)
                if self._norm_commutator < tmp:
                    self._norm_commutator = tmp

                tmp = np.linalg.norm(g_mat)
                if self._norm_grad < tmp:
                    self._norm_grad = tmp

        return e_total + self.e_sic, g_vec_u

    def update_ks_energy(self, ham, wfs, dens):
        """
        Update Kohn-Sham energy
        It assumes the temperature is zero K.
        """

        dens.update(wfs)
        ham.update(dens, wfs, False)

        return ham.get_energy(0.0, wfs, False)

    def evaluate_phi_and_der_phi(self, a_vec_u, p_mat_u, n_dim, alpha,
                                 ham, wfs, dens, c_ref,
                                 phi=None, g_vec_u=None):
        """
        phi = f(x_k + alpha_k*p_k)
        der_phi = \\grad f(x_k + alpha_k*p_k) \\cdot p_k
        :return:  phi, der_phi # floats
        """
        if phi is None or g_vec_u is None:
            x_mat_u = {k: a_vec_u[k] + alpha * p_mat_u[k] for k in a_vec_u}
            phi, g_vec_u = self.get_energy_and_gradients(x_mat_u, n_dim,
                                                         ham, wfs, dens, c_ref)

            # If GMF is used save the original gradient and invert the parallel
            # projection onto the eigenvectors with negative eigenvalues
            if self.gmf:
                self.g_vec_u_original = deepcopy(g_vec_u)
                g_vec_u = self.searchdir_algo.invert_parallel_grad(g_vec_u)

        der_phi = 0.0
        for k in p_mat_u:
            der_phi += g_vec_u[k].conj() @ p_mat_u[k]

        der_phi = der_phi.real
        der_phi = wfs.kd.comm.sum(der_phi)

        return phi, der_phi, g_vec_u

    def update_ref_orbitals(self, wfs, ham, dens):
        """
        Update reference orbitals

        :param wfs:
        :param ham:
        :return:
        """

        if self.representation == 'full':
            badgrad = self._norm_commutator > self._norm_grad / 3. and \
                self.checkgraderror
        else:
            badgrad = False
        counter = self.update_ref_orbs_counter
        if (self.iters % counter == 0 and self.iters > 1) or \
                (self.restart and self.iters > 1) or badgrad:
            self.iters = 1
            if self.update_ref_orbs_canonical or self.restart:
                self.get_canonical_representation(ham, wfs, dens)
            else:
                self.dm_helper.set_reference_orbitals(wfs, self.n_dim)
                for kpt in wfs.kpt_u:
                    u = self.kpointval(kpt)
                    self.a_vec_u[u] = np.zeros_like(self.a_vec_u[u])

            # Erase memory of search direction algorithm
            self.searchdir_algo.reset()

    def get_preconditioning(self, wfs, use_prec, make_pd=False):

        if not use_prec:
            return None

        if self.searchdir_algo.name == 'l-bfgs-p':
            beta0 = self.searchdir_algo.beta_0
            gamma = 0.25
        else:
            beta0 = 1.0
            gamma = 0.0

        counter = self.update_precond_counter
        precond = {}
        for kpt in wfs.kpt_u:
            k = self.kpointval(kpt)
            w = kpt.weight / (3.0 - wfs.nspins)
            if self.iters % counter == 0 or self.iters == 1:
                self.hess[k] = self.get_hessian(kpt)
                if make_pd:
                    if self.dtype == float:
                        self.hess[k] = np.abs(self.hess[k])
                    else:
                        self.hess[k] = np.abs(self.hess[k].real) \
                            + 1.0j * np.abs(self.hess[k].imag)
            hess = self.hess[k]
            precond[k] = np.zeros_like(hess)
            correction = w * gamma * beta0 ** (-1)
            if self.searchdir_algo.name != 'l-bfgs-p':
                correction = np.zeros_like(hess)
                zeros = abs(hess) < 1.0e-4
                correction[zeros] = 1.0
            precond[k] += 1. / ((1 - gamma) * hess.real + correction)
            if self.dtype == complex:
                precond[k] += 1.j / ((1 - gamma) * hess.imag + correction)

        return precond

    def get_hessian(self, kpt):
        """
        Calculate the following diagonal approximation to the Hessian:
        h_{lm, lm} = -2.0 * (eps_n[l] - eps_n[m]) * (f[l] - f[m])
        """

        f_n = kpt.f_n
        eps_n = kpt.eps_n
        u = self.kpointval(kpt)
        il1 = list(self.ind_up[u])

        hess = np.zeros(len(il1[0]), dtype=self.dtype)
        x = 0
        for n, m in zip(*il1):
            df = f_n[n] - f_n[m]
            hess[x] = -2.0 * (eps_n[n] - eps_n[m]) * df
            if abs(hess[x]) < 1.0e-10:
                hess[x] = 0.0
            if self.dtype == complex:
                hess[x] += 1.0j * hess[x]
            x += 1

        return hess

    def get_canonical_representation(self, ham, wfs, dens,
                                     sort_eigenvalues=False):
        """
        Choose canonical orbitals as the orbitals that diagonalize the
        Lagrange matrix. It is probably necessary to do a subspace rotation
        with equally occupied orbitals as the total energy is unitary invariant
        within equally occupied subspaces.
        """

        with wfs.timer('Get canonical representation'):
            for kpt in wfs.kpt_u:
                self.dm_helper.update_to_canonical_orbitals(
                    wfs, ham, kpt, self.update_ref_orbs_canonical,
                    self.restart)

            self._e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            occ_name = getattr(wfs.occupations, "name", None)
            if occ_name == 'mom':
                if not sort_eigenvalues:
                    self.sort_orbitals_mom(wfs)
                else:
                    self.sort_orbitals(ham, wfs, use_eps=True)
                    not_update = not wfs.occupations.update_numbers
                    fixed_occ = wfs.occupations.use_fixed_occupations
                    if not_update or fixed_occ:
                        wfs.occupations.numbers = \
                            self.initial_occupation_numbers

            self.dm_helper.set_reference_orbitals(wfs, self.n_dim)
            for kpt in wfs.kpt_u:
                u = self.kpointval(kpt)
                self.a_vec_u[u] = np.zeros_like(self.a_vec_u[u])

    def reset(self):

        self.dm_helper = None
        self.error = np.inf
        self.initialized = False
        self.searchdir_algo.reset()

    def sort_orbitals(self, ham, wfs, use_eps=False):
        """
        Sort orbitals according to the eigenvalues or
        the diagonal elements of the Hamiltonian matrix
        """

        with wfs.timer('Sort WFS'):
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                if use_eps:
                    orbital_energies = kpt.eps_n
                else:
                    orbital_energies = self.dm_helper.orbital_energies(
                        wfs, ham, kpt)
                ind = np.argsort(orbital_energies)
                # check if it is necessary to sort
                x = np.max(abs(ind - np.arange(len(ind))))

                if x > 0:
                    # now sort wfs according to orbital energies
                    self.dm_helper.sort_orbitals(wfs, kpt, ind)
                    kpt.f_n[np.arange(len(ind))] = kpt.f_n[ind]
                    kpt.eps_n[np.arange(len(ind))] = orbital_energies[ind]
                    occ_name = getattr(wfs.occupations, "name", None)
                    if occ_name == 'mom':
                        # OccupationsMOM.numbers needs to be updated after
                        # sorting
                        self.update_mom_numbers(wfs, kpt)
                    if self.constraints:
                        # Identity of the contrained orbitals has changed and
                        # needs to be updated
                        self.constraints[k] = update_constraints(
                            self.constraints[k], list(ind))

    def sort_orbitals_mom(self, wfs):
        """
        Sort orbitals according to the occupation
        numbers so that there are no holes in the
        distribution of occupation numbers
        :return:
        """
        changedocc = False
        for kpt in wfs.kpt_u:
            k = self.kpointval(kpt)
            occupied = kpt.f_n > 1.0e-10
            n_occ = len(kpt.f_n[occupied])
            if n_occ == 0.0:
                continue
            if np.min(kpt.f_n[:n_occ]) == 0:
                ind_occ = np.argwhere(occupied)
                ind_unocc = np.argwhere(~occupied)
                ind = np.vstack((ind_occ, ind_unocc))
                ind = np.squeeze(ind)
                self.dm_helper.sort_orbitals(wfs, kpt, ind)
                kpt.f_n = kpt.f_n[ind]
                kpt.eps_n = kpt.eps_n[ind]

                # OccupationsMOM.numbers needs to be updated after sorting
                self.update_mom_numbers(wfs, kpt)

                if self.constraints:
                    # Identities of the contrained orbitals have changed and
                    # needs to be updated
                    self.constraints[k] = update_constraints(
                        self.constraints[k], list(ind))

                changedocc = True

        return changedocc

    def todict(self):

        ret = {'name': self.name,
               'searchdir_algo': self.searchdir_algo.todict(),
               'linesearch_algo': self.line_search.todict(),
               'localizationtype': self.localizationtype,
               'update_ref_orbs_counter': self.update_ref_orbs_counter,
               'update_precond_counter': self.update_precond_counter,
               'use_prec': self.use_prec,
               'matrix_exp': self.matrix_exp,
               'representation': self.representation,
               'functional': self.func.todict(),
               'orthonormalization': self.orthonormalization,
               'constraints': self.constraints
               }
        if self.gmf:
            ret['partial_diagonalizer'] = \
                self.searchdir_algo.partial_diagonalizer.todict()
        return ret

    def rotate_wavefunctions(self, wfs, a_vec_u, n_dim, c_ref):

        """
        Apply unitary transformation U = exp(A) to
        the orbitals c_ref

        :param wfs:
        :param a_vec_u:
        :param n_dim:
        :param c_ref:
        :return:
        """

        with wfs.timer('Unitary rotation'):
            for kpt in wfs.kpt_u:
                k = self.kpointval(kpt)
                if n_dim[k] == 0:
                    continue

                u_nn = self.get_exponential_matrix_kpt(wfs, kpt,
                                                       a_vec_u,
                                                       n_dim)

                self.dm_helper.appy_transformation_kpt(
                    wfs, u_nn.T, kpt, c_ref[k], False, False)

                with wfs.timer('Calculate projections'):
                    self.dm_helper.update_projections(wfs, kpt)

    def get_exponential_matrix_kpt(self, wfs, kpt, a_vec_u, n_dim):
        """
        Get unitary matrix U as the exponential of a skew-Hermitian
        matrix A (U = exp(A))
        """

        k = self.kpointval(kpt)

        if self.gd.comm.rank == 0:
            if self.matrix_exp == 'egdecomp-u-invar' and \
                    self.representation == 'u-invar':
                n_occ = get_n_occ(kpt)
                n_v = n_dim[k] - n_occ
                a_mat = a_vec_u[k].reshape(n_occ, n_v)
            else:
                a_mat = vec2skewmat(a_vec_u[k], n_dim[k],
                                    self.ind_up[k], self.dtype)

            if self.matrix_exp == 'pade-approx':
                # this function takes a lot of memory
                # for large matrices... what can we do?
                with wfs.timer('Pade Approximants'):
                    u_nn = np.ascontiguousarray(expm(a_mat))
            elif self.matrix_exp == 'egdecomp':
                # this method is based on diagonalization
                with wfs.timer('Eigendecomposition'):
                    u_nn, evecs, evals = expm_ed(a_mat, evalevec=True)
            elif self.matrix_exp == 'egdecomp-u-invar':
                with wfs.timer('Eigendecomposition'):
                    u_nn = expm_ed_unit_inv(a_mat, oo_vo_blockonly=False)

        with wfs.timer('Broadcast u_nn'):
            if self.gd.comm.rank != 0:
                u_nn = np.zeros(shape=(n_dim[k], n_dim[k]),
                                dtype=wfs.dtype)
            self.gd.comm.broadcast(u_nn, 0)

        if self.matrix_exp == 'egdecomp':
            with wfs.timer('Broadcast evecs and evals'):
                if self.gd.comm.rank != 0:
                    evecs = np.zeros(shape=(n_dim[k], n_dim[k]),
                                     dtype=complex)
                    evals = np.zeros(shape=n_dim[k],
                                     dtype=float)
                self.gd.comm.broadcast(evecs, 0)
                self.gd.comm.broadcast(evals, 0)
                self.evecs[k], self.evals[k] = evecs, evals

        return u_nn

    def check_assertions(self, wfs, dens):

        assert dens.mixer.driver.basemixerclass.name == 'no-mixing', \
            'Please, use: mixer={\'backend\': \'no-mixing\'}'
        if wfs.occupations.name != 'mom':
            errormsg = \
                'Please, use occupations={\'name\': \'fixed-uniform\'}'
            assert wfs.occupations.name == 'fixed-uniform', errormsg

    def initialize_mom(self, wfs, dens):
        # Reinitialize the MOM reference orbitals
        # after orthogonalization/localization
        wfs.occupations.initialize_reference_orbitals()
        wfs.calculate_occupation_numbers(dens.fixed)
        self.sort_orbitals_mom(wfs)

    def check_mom(self, wfs, dens):
        occ_name = getattr(wfs.occupations, "name", None)
        if occ_name == 'mom':
            self._e_entropy = wfs.calculate_occupation_numbers(dens.fixed)
            self.restart = self.sort_orbitals_mom(wfs)

    def update_mom_numbers(self, wfs, kpt):
        if wfs.collinear and wfs.nspins == 1:
            degeneracy = 2
        else:
            degeneracy = 1
        wfs.occupations.numbers[kpt.s] = \
            kpt.f_n / (kpt.weightk * degeneracy)

    def randomize_orbitals_kpt(self, wfs, kpt):
        """
        Add random noise to orbitals but keep them orthonormal
        """
        nst = self.nbands
        wt = kpt.weight * 0.01
        arand = wt * random_a((nst, nst), wfs.dtype)
        arand = arand - arand.T.conj()
        wfs.gd.comm.broadcast(arand, 0)
        self.dm_helper.appy_transformation_kpt(wfs, expm(arand), kpt)

    def calculate_hamiltonian_matrix(self, hamiltonian, wfs, kpt):
        H_MM = self.dm_helper.calculate_hamiltonian_matrix(
            hamiltonian, wfs, kpt)
        return H_MM

    def kpointval(self, kpt):
        return self.nkpts * kpt.s + kpt.q

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, e):
        self._error = e


def get_n_occ(kpt):
    """
    Get number of occupied orbitals
    """
    return len(kpt.f_n) - np.searchsorted(kpt.f_n[::-1], 1e-10)


def random_a(shape, dtype):

    a = np.random.random_sample(shape)
    if dtype == complex:
        a = a.astype(complex)
        a += 1.0j * np.random.random_sample(shape)

    return a


def vec2skewmat(a_vec, dim, ind_up, dtype):

    a_mat = np.zeros(shape=(dim, dim), dtype=dtype)
    a_mat[ind_up] = a_vec
    a_mat -= a_mat.T.conj()
    np.fill_diagonal(a_mat, a_mat.diagonal() * 0.5)
    return a_mat


def convert_constraints(constraints, n_dim, n_occ, representation):
    """
    Parses and checks the user input of constraints. If constraints are passed
    as a list of a single orbital index all pairs of orbitals involving this
    index are added to the constraints.

    :param constraints: List of constraints for one K-point
    :param n_dim:
    :param n_occ:
    :param representation: Unitary invariant, sparse or full representation
                           determining the electronic degrees of freedom that
                           need to be constrained

    :return: Converted list of constraints
    """

    new = constraints.copy()
    for con in constraints:
        assert isinstance(con, list) or isinstance(con, int), \
            'Check constraints.'
        if isinstance(con, list):
            assert len(con) < 3, 'Check constraints.'
            if len(con) == 1:
                con = con[0]  # List of single index
            else:
                # Make first index always smaller than second index
                if representation != 'full' and con[1] < con[0]:
                    temp = deepcopy(con[0])
                    con[0] = deepcopy(con[1])
                    con[1] = temp
                check_indices(
                    con[0], con[1], n_dim, n_occ, representation)
                continue
        # Add all pairs containing the single index
        if isinstance(con, int):
            new += find_all_pairs(con, n_dim, n_occ, representation)

    # Delete all list containing a single index
    done = False
    while not done:
        done = True
        for i in range(len(new)):
            if len(new[i]) < 2:
                del new[i]
                done = False
                break
    return new


def check_indices(ind1, ind2, n_dim, n_occ, representation):
    """
    Makes sure the user input for the constraints makes sense.
    """

    assert ind1 != ind2, 'Check constraints.'
    if representation == 'full':
        assert ind1 < n_dim and ind2 < n_dim, 'Check constraints.'
    elif representation == 'sparse':
        assert ind1 < n_occ and ind2 < n_dim, 'Check constraints.'
    elif representation == 'u-invar':
        assert ind1 < n_occ and ind2 >= n_occ and ind2 < n_dim, \
            'Check constraints.'


def find_all_pairs(ind, n_dim, n_occ, representation):
    """
    Creates a list of all orbital pairs corresponding to degrees of freedom of
    the system containing an orbital index.

    :param ind: The orbital index
    :param n_dim:
    :param n_occ:
    :param representation: Unitary invariant, sparse or full, defining what
                           index pairs correspond to degrees of freedom of the
                           system
    """

    pairs = []
    if representation == 'u-invar':
        # Only ov rotations are degrees of freedom
        if ind < n_occ:
            for i in range(n_occ, n_dim):
                pairs.append([ind, i])
        else:
            for i in range(n_occ):
                pairs.append([i, ind])
    else:
        if (ind < n_occ and representation == 'sparse') \
                or representation == 'full':
            # oo and ov rotations are degrees of freedom
            for i in range(n_dim):
                if i == ind:
                    continue
                pairs.append([i, ind] if i < ind else [ind, i])
                if representation == 'full':
                    # The orbital rotation matrix is not assumed to be
                    # antihermitian, so the reverse order of indices must be
                    # added as a second constraint
                    pairs.append([ind, i] if i < ind else [i, ind])
        else:
            for i in range(n_occ):
                pairs.append([i, ind])
    return pairs


def update_constraints(constraints, ind):
    """
    Change the constraint indices to match a new indexation, e.g. due to
    sorting the orbitals

    :param constraints: The list of constraints for one K-point
    :param ind: List containing information about the change in indexation
    """

    new = deepcopy(constraints)
    for i in range(len(constraints)):
        for k in range(len(constraints[i])):
            new[i][k] = ind.index(constraints[i][k])
    return new
