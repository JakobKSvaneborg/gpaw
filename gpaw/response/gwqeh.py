from __future__ import division, print_function

import sys
from math import pi
import pickle

import numpy as np

from ase.units import Hartree, Bohr
from ase.dft.kpoints import monkhorst_pack

import gpaw.mpi as mpi
from gpaw.old.kpt_descriptor import KPointDescriptor
from gpaw.response.hilbert import HilbertTransform
from gpaw.response.g0w0 import select_kpts
from gpaw.response.groundstate import ResponseGroundStateAdapter
from gpaw.response.context import ResponseContext
from gpaw.response.pair import (KPointPairFactory, ActualPairDensityCalculator,
                                phase_shifted_fft_indices)
from gpaw.response.qpd import SingleQPWDescriptor


def frequency_grid(domega0, omega2, omegamax):
    beta = (2**0.5 - 1) * domega0 / omega2
    wmax = int(omegamax / (domega0 + beta * omegamax)) + 2
    w = np.arange(wmax)
    omega_w = w * domega0 / (1 - beta * w)
    return omega_w


# Hard-coded toggle: when True, GWQEHCorrection.calculate_W_QEH routes to
# the legacy single-basis qeh.old_qeh.Heterostructure path instead of the
# modern multi-basis qeh.QEH wrapper. Used internally to reproduce
# previously published GWQEH numbers against the same screening engine
# they were originally generated with. Not exposed as a public knob.
_USE_LEGACY_QEH = False


class GWQEHCorrection:
    def __init__(self, calc, gwfile=None, filename=None, kpts=[0], bands=None,
                 structure=None, d=None, layer=0,
                 dW_qw=None, qqeh=None, wqeh=None,
                 txt=sys.stdout, world=mpi.world, domega0=0.025,
                 omega2=10.0, eta=0.1, include_q0=True, metal=False,
                 restart=False):
        """
        Class for calculating quasiparticle energies of van der Waals
        heterostructures using the GW approximation for the self-energy.
        The quasiparticle energy correction due to increased screening from
        surrounding layers is obtained from the QEH model.
        Parameters:

        calc: str or PAW object
            GPAW calculator object or filename of saved calculator object.
        gwfile: str or None
            name of gw results file from the monolayer calculation
        filename: str
            filename for gwqeh output
        kpts: list
            List of indices of sthe IBZ k-points to calculate the quasi
            particle energies for. Set to [0] by default since the QP
            correction is generally the same for all k.
        bands: tuple
            Range of band indices, like (n1, n2+1), to calculate the quasi
            particle energies for. Note that the second band index is not
            included. Should be the same as used for the GW calculation.
        structure: list of str
            Heterostructure set up. Each entry should consist of number of
            layers + chemical formula.
            For example: ['3H-MoS2', graphene', '10H-WS2'] gives 3 layers of
            H-MoS2, 1 layer of graphene and 10 layers of H-WS2.
            The name of the layers should correspond to building block files:
            "<name>-chi.npz" in the local repository.
        d: array of floats
            Interlayer distances for neighboring layers in Ang.
            Length of array = number of layers - 1
            OR
            layerwidth_n or layerwidth_l as documented in QEH
        layer: int
            index of layer to calculate QP correction for.
        dW_qw: 2D array of floats dimension q X w
            Change in screened interaction. Should be set to None to calculate
            dW directly from buildingblocks.
        qqeh: array of floats
            q-grid used for dW_qw (only needed if dW is given by hand).
        wqeh: array of floats
            w-grid used for dW_qw. So far this have to be the same as for the
            GWQEH calculation.  (only needed if dW is given by hand).
        domega0: float
            Minimum frequency step (in eV) used in the generation of the non-
            linear frequency grid.
        omega2: float
            Control parameter for the non-linear frequency grid, equal to the
            frequency where the grid spacing has doubled in size.
        eta: float
            Broadening parameter.
        include_q0: bool
            include q=0 in W or not. if True an integral arround q=0 is
            performed, if False the q=0 contribution is set to zero.
        metal: bool
            If True, the point at q=0 is omitted when averaging the screened
            potential close to q=0.
        """
        self.restart = restart
        self.gwfile = gwfile

        self.inputcalc = calc
        self.gs = ResponseGroundStateAdapter.from_input(calc)
        context_txt = filename + '.txt' if filename is not None else txt
        self.context = ResponseContext(txt=context_txt, comm=world)
        self.world = world

        # Initialize parallelization communicators
        # Assuming nblocks=1 as per original code behavior
        self.blockcomm = world.new_communicator([world.rank])
        self.kncomm = world

        # Set low ecut in order to use PairDensity object since only
        # G=0 is needed.
        self.ecut = 0.1

        self.kptpair_factory = KPointPairFactory(self.gs, self.context)
        self.pair_calc = ActualPairDensityCalculator(self.kptpair_factory,
                                                     self.blockcomm)

        if txt == sys.stdout:
            self.fd = sys.stdout
        else:
            self.fd = self.context.fd

        self.filename = filename
        self.ecut /= Hartree
        self.eta = eta / Hartree
        self.domega0 = domega0 / Hartree
        self.omega2 = omega2 / Hartree

        self.kpts = list(select_kpts(kpts, self.gs.kd))

        self.nocc2 = self.gs.nocc2
        if bands is None:
            bands = [0, self.nocc2]

        self.bands = bands

        b1, b2 = bands
        self.shape = shape = (self.gs.nspins, len(self.kpts), b2 - b1)
        self.eps_sin = np.empty(shape)     # KS-eigenvalues
        self.f_sin = np.empty(shape)       # occupation numbers
        self.sigma_sin = np.zeros(shape)   # self-energies
        self.dsigma_sin = np.zeros(shape)  # derivatives of self-energies
        self.Z_sin = None                  # renormalization factors
        self.qp_sin = None
        self.Qp_sin = None

        self.ecutnb = 150 / Hartree
        vol = abs(np.linalg.det(self.gs.gd.cell_cv))
        self.vol = vol
        # get_number_of_bands is typically nbands in gd
        self.nbands = min(self.gs.bd.nbands,
                          int(vol * (self.ecutnb)**1.5 * 2**0.5 / 3 / pi**2))

        self.nspins = self.gs.nspins

        kd = self.gs.kd

        self.mysKn1n2 = None  # my (s, K, n1, n2) indices
        self.distribute_k_points_and_bands(b1, b2, kd.ibz2bz_k[self.kpts])

        # Find q-vectors and weights in the IBZ:
        assert -1 not in kd.bz2bz_ks
        offset_c = 0.5 * ((kd.N_c + 1) % 2) / kd.N_c
        bzq_qc = monkhorst_pack(kd.N_c) + offset_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.gs.atoms, kd.symmetry)

        # frequency grid
        omax = self.find_maximum_frequency()
        self.omega_w = frequency_grid(self.domega0, self.omega2, omax)
        self.nw = len(self.omega_w)
        self.wsize = 2 * self.nw

        # Install the screened-potential difference dW. Default impl
        # below handles the (nq, nw) scalar / monopole representation;
        # subclasses (e.g. GWmQEHCorrection) override to install a
        # matrix representation instead.
        self._setup_dW(dW_qw=dW_qw, qqeh=qqeh, wqeh=wqeh,
                       structure=structure, d=d, layer=layer,
                       restart=restart,
                       include_q0=include_q0, metal=metal)

        self.htp = HilbertTransform(self.omega_w, self.eta, gw=True)
        self.htm = HilbertTransform(self.omega_w, -self.eta, gw=True)

        self.complete = False
        self.nq = 0
        if self.load_state_file():
            if self.complete:
                print('Self-energy loaded from file', file=self.fd)

        print('Initialized GWQEHCorrection object', file=self.fd)

    def distribute_k_points_and_bands(self, band1, band2, kpts=None):
        """Distribute spins, k-points and bands."""
        if kpts is None:
            kpts = np.arange(self.gs.kd.nbzkpts)

        nbands = band2 - band1
        size = self.kncomm.size
        rank = self.kncomm.rank
        ns = self.gs.nspins
        nk = len(kpts)
        n = (ns * nk * nbands + size - 1) // size
        i1 = rank * n
        i2 = min(i1 + n, ns * nk * nbands)

        self.mysKn1n2 = []
        i = 0
        for s in range(ns):
            for K in kpts:
                n1 = min(max(0, i1 - i), nbands)
                n2 = min(max(0, i2 - i), nbands)
                if n1 != n2:
                    self.mysKn1n2.append((s, K, n1 + band1, n2 + band1))
                i += nbands

        print('BZ k-points:', self.gs.kd, file=self.fd)
        print('Distributing spins, k-points and bands (%d x %d x %d)' %
              (ns, nk, nbands),
              'over %d process%s' %
              (self.kncomm.size, ['es', ''][self.kncomm.size == 1]),
              file=self.fd)
        print('Number of blocks:', self.blockcomm.size, file=self.fd)

    def _setup_dW(self, *, dW_qw, qqeh, wqeh, structure, d, layer,
                  restart, include_q0, metal):
        """Initialize dW from input, npz, or QEH and install it on the
        GW grid.

        Default implementation: scalar (nq, nw) monopole stored as
        ``self.dW_qw``. Subclasses override to install a different
        representation (e.g. the mQEH matrix).
        """
        if dW_qw is None:
            if restart:
                try:
                    data = np.load(self.filename + "_dW_qw.npz")
                    self.qqeh = data['qqeh']
                    self.wqeh = data['wqeh']
                    dW_qw = data['dW_qw']
                except IOError:
                    dW_qw = self.calculate_W_QEH(structure, d, layer)
            else:
                dW_qw = self.calculate_W_QEH(structure, d, layer)
        else:
            self.qqeh = qqeh
            self.wqeh = wqeh

        self.dW_qw = self.get_W_on_grid(dW_qw, include_q0=include_q0,
                                        metal=metal)
        assert self.nw == self.dW_qw.shape[1], \
            'Frequency grids do not match!'

    def calculate_QEH(self):
        print('Calculating QEH self-energy contribution', file=self.fd)

        kd = self.gs.kd
        # Use ResponseGroundStateAdapter's atomrotations
        atomrotations = self.gs.atomrotations

        # Reset calculation
        self.sigma_sin = np.zeros(self.shape)   # self-energies
        self.dsigma_sin = np.zeros(self.shape)  # derivatives of self-energies

        # Get KS eigenvalues and occupation numbers:
        b1, b2 = self.bands
        for i, k in enumerate(self.kpts):
            for s in range(self.nspins):
                kpt = self.gs.kpt_ks[k][s]
                self.eps_sin[s, i] = kpt.eps_n[b1:b2]
                self.f_sin[s, i] = kpt.f_n[b1:b2] / kpt.weight

        # My part of the states we want to calculate QP-energies for:
        # Use KPointPairFactory to get KPoints
        mykpts = [self.kptpair_factory.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in self.mysKn1n2]

        Nq = len((self.qd.ibzk_kc))
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            self.nq = iq
            nq = iq
            self.save_state_file()

            qcstr = '(' + ', '.join(['%.3f' % x for x in q_c]) + ')'
            print('Calculating contribution from IBZ q-point #%d/%d q_c=%s'
                  % (nq, Nq, qcstr), file=self.fd)

            # Screened potential. QEH returns dW in Hartree*Bohr^2; the
            # factor L absorbs the 1/L in x = 1/(N_q*2pi*Omega) so that
            # x*L = 1/(N_q*2pi*A) matches Eq.(9) of W&T 2017.
            dW_w = self.dW_qw[nq]
            dW_w = dW_w[:, np.newaxis, np.newaxis]
            L = abs(self.gs.gd.cell_cv[2, 2])
            dW_w *= L

            nw = self.nw

            Wpm_w = np.zeros([2 * nw, 1, 1], dtype=complex)
            Wpm_w[:nw] = dW_w
            Wpm_w[nw:] = Wpm_w[0:nw]

            self.htp(Wpm_w[:nw])
            self.htm(Wpm_w[nw:])

            # Setup q-point descriptor
            pd0 = SingleQPWDescriptor.from_q(
                q_c, self.ecut, self.gs.gd, gammacentered=True)
            G_Gv = pd0.get_reciprocal_vectors()
            assert len(G_Gv) == 1
            assert np.allclose(pd0.get_reciprocal_vectors(add_q=False), 0)

            # Initialize PAW corrections
            self.Q_aGii = self.gs.pair_density_paw_corrections(pd0).Q_aGii

            # Loop over all k-points in the BZ and find those that are related
            # to the current IBZ k-point by symmetry
            Q1 = self.qd.ibz2bz_k[iq]
            Q2s = set()
            for s, Q2 in enumerate(self.qd.bz2bz_ks[Q1]):
                if Q2 >= 0 and Q2 not in Q2s:
                    Q2s.add(Q2)
            for Q2 in Q2s:
                s = self.qd.sym_k[Q2]
                self.s = s
                U_cc = self.qd.symmetry.op_scc[s]
                time_reversal = self.qd.time_reversal_k[Q2]
                self.sign = 1 - 2 * time_reversal
                Q_c = self.qd.bzk_kc[Q2]
                d_c = self.sign * np.dot(U_cc, q_c) - Q_c
                assert np.allclose(d_c.round(), d_c)

                for u1, kpt1 in enumerate(mykpts):
                    K2 = kd.find_k_plus_q(Q_c, [kpt1.K])[0]
                    # Get kpt2 using factory, blockcomm for parallellization
                    kpt2 = self.kptpair_factory.get_k_point(
                        kpt1.s, K2, 0, self.nbands, blockcomm=self.blockcomm)
                    k1 = kd.bz2ibz_k[kpt1.K]
                    i = self.kpts.index(k1)

                    # Determine FFT indices
                    def coordinate_transformation(q_c):
                        return self.sign * np.dot(U_cc, q_c)

                    I_G = phase_shifted_fft_indices(
                        kpt1.k_c, kpt2.k_c, pd0,
                        coordinate_transformation=coordinate_transformation)

                    pos_av = self.gs.get_pos_av()
                    M_vv = np.dot(self.gs.gd.cell_cv.T,
                                  np.dot(U_cc.T,
                                         np.linalg.inv(self.gs.gd.cell_cv).T))
                    Q_aGii = []
                    for a, Q_Gii in enumerate(self.Q_aGii):
                        x_G = np.exp(1j * np.dot(G_Gv, (pos_av[a] -
                                                        np.dot(M_vv,
                                                               pos_av[a]))))
                        R_sii = atomrotations.get_R_asii()[a]
                        U_ii = R_sii[self.s]
                        Q_Gii = np.dot(np.dot(U_ii, Q_Gii * x_G[:, None,
                                                                None]),
                                       U_ii.T).transpose(1, 0, 2)
                        if self.sign == -1:
                            Q_Gii = Q_Gii.conj()
                        Q_aGii.append(Q_Gii)

                    for n in range(kpt1.n2 - kpt1.n1):
                        ut1cc_R = kpt1.ut_nR[n].conj()
                        eps1 = kpt1.eps_n[n]
                        # PAW correction application
                        C1_aGi = [np.dot(Qa_Gii, P1_ni[n].conj())
                                  for Qa_Gii, P1_ni in zip(Q_aGii, kpt1.P_ani)]

                        # Calculate pair density
                        n_mG = self.pair_calc.calculate_pair_density(
                            ut1cc_R, C1_aGi, kpt2, pd0, I_G)

                        if self.sign == 1:
                            n_mG = n_mG.conj()

                        f_m = kpt2.f_n
                        deps_m = eps1 - kpt2.eps_n
                        sigma, dsigma = self.calculate_sigma(n_mG, deps_m,
                                                             f_m, Wpm_w)
                        nn = kpt1.n1 + n - self.bands[0]
                        self.sigma_sin[kpt1.s, i, nn] += sigma
                        self.dsigma_sin[kpt1.s, i, nn] += dsigma

        self.world.sum(self.sigma_sin)
        self.world.sum(self.dsigma_sin)

        self.complete = True
        self.save_state_file()

        return self.sigma_sin, self.dsigma_sin

    def calculate_qp_correction(self):

        if self.complete:
            print('Self-energy loaded from file', file=self.fd)
        else:
            self.calculate_QEH()

        # Need GW result for renormalization factor
        b1, b2 = self.bands
        if self.gwfile is not None:
            gwdata = pickle.load(open(self.gwfile, 'rb'))

            self.dsigmagw_sin = gwdata['dsigma']
            self.qpgw_sin = gwdata['qp'] / Hartree

            nk = self.qpgw_sin.shape[1]
            if not self.sigma_sin.shape[1] == nk:
                self.sigma_sin = np.repeat(
                    self.sigma_sin[:, :1, :], nk, axis=1)
                self.dsigma_sin = np.repeat(
                    self.dsigma_sin[:, :1, :], nk, axis=1)
            self.Z_sin = 1. / (1 - self.dsigma_sin - self.dsigmagw_sin)
        else:
            # Z = 0.7 is a good estimate according to
            # https://doi.org/10.1038/s41524-020-00480-7
            print('estimating quasiparticle weight Z = 0.7')
            self.Z_sin = 0.7
        self.qp_sin = self.Z_sin * self.sigma_sin

        return self.qp_sin * Hartree

    def calculate_qp_energies(self):
        # calculate
        assert self.gwfile is not None, \
            'gwfile must be specified to calculate qp energies!'
        qp_sin = self.calculate_qp_correction() / Hartree
        self.Qp_sin = self.qpgw_sin + qp_sin
        self.save_state_file()
        return self.Qp_sin * Hartree

    def calculate_sigma(self, n_mG, deps_m, f_m, W_wGG):
        """Calculates a contribution to the self-energy and its derivative for
        a given (k, k-q)-pair from its corresponding pair-density and
        energy."""
        o_m = abs(deps_m)
        # Add small number to avoid zeros for degenerate states:
        sgn_m = np.sign(deps_m + 1e-15)

        # Pick +i*eta or -i*eta:
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2
        comm = self.blockcomm
        nw = len(self.omega_w)
        nG = n_mG.shape[1]
        mynG = (nG + comm.size - 1) // comm.size
        Ga = min(comm.rank * mynG, nG)
        Gb = min(Ga + mynG, nG)
        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        o1_m = self.omega_w[w_m]
        o2_m = self.omega_w[w_m + 1]
        x = 1.0 / (self.qd.nbzkpts * 2 * pi * self.vol)
        sigma = 0.0
        dsigma = 0.0

        # Performing frequency integration
        for o, o1, o2, sgn, s, w, n_G in zip(o_m, o1_m, o2_m,
                                             sgn_m, s_m, w_m, n_mG):

            C1_GG = W_wGG[s * nw + w]
            C2_GG = W_wGG[s * nw + w + 1]
            p = x * sgn
            myn_G = n_G[Ga:Gb]
            sigma1 = p * np.dot(np.dot(myn_G, C1_GG), n_G.conj()).imag
            sigma2 = p * np.dot(np.dot(myn_G, C2_GG), n_G.conj()).imag
            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)

        return sigma, dsigma

    def save_state_file(self, q=0):
        data = {'kpts': self.kpts,
                'bands': self.bands,
                'nbands': self.nbands,
                'last_q': self.nq,
                'complete': self.complete,
                'sigma_sin': self.sigma_sin,
                'dsigma_sin': self.dsigma_sin,
                'qp_sin': self.qp_sin,
                'Qp_sin': self.Qp_sin}
        if self.world.rank == 0:
            np.savez(self.filename + '_qeh.npz',
                     **data)

    def load_state_file(self):
        if not self.restart:
            return False
        try:
            data = np.load(self.filename + '_qeh.npz')
        except IOError:
            return False
        else:
            # Use np.array_equal so multi-kpt arrays don't raise
            # "truth value of an array is ambiguous" under the and chain.
            if (np.array_equal(data['kpts'], self.kpts)
                    and np.array_equal(data['bands'], self.bands)
                    and data['nbands'] == self.nbands):
                self.nq = data['last_q']
                self.complete = data['complete']
                self.sigma_sin = data['sigma_sin']
                self.dsigma_sin = data['dsigma_sin']
                return True
            else:
                return False

    def get_W_on_grid(self, dW_qw, include_q0=True, metal=False):
        """This function transforms the screened potential W(q,w) to the
        (q,w)-grid of the GW calculation. Also, W is integrated over
        a region around q=0 if include_q0 is set to True."""

        q_cs = self.qd.ibzk_kc

        rcell_cv = 2 * pi * np.linalg.inv(self.gs.gd.cell_cv).T
        q_vs = np.dot(q_cs, rcell_cv)
        q_grid = (q_vs**2).sum(axis=1)**0.5
        self.q_grid = q_grid
        w_grid = self.omega_w

        wqeh = self.wqeh  # w_grid.copy() # self.qeh
        qqeh = self.qqeh
        sortqeh = np.argsort(qqeh)
        qqeh = qqeh[sortqeh]
        dW_qw = dW_qw[sortqeh]

        sort = np.argsort(q_grid)
        isort = np.argsort(sort)
        if metal and np.isclose(qqeh[0], 0):
            """We don't have the right q=0 limit for metals  and semi-metals.
            -> Point should be omitted from interpolation"""
            qqeh = qqeh[1:]
            dW_qw = dW_qw[1:]
            sort = sort[1:]

        from scipy.interpolate import RectBivariateSpline
        yr = RectBivariateSpline(qqeh, wqeh, dW_qw.real, s=0)
        yi = RectBivariateSpline(qqeh, wqeh, dW_qw.imag, s=0)

        dWgw_qw = yr(q_grid[sort], w_grid) + 1j * yi(q_grid[sort], w_grid)
        dW_qw = yr(qqeh, w_grid) + 1j * yi(qqeh, w_grid)

        if metal:
            # Interpolation is done -> put back zeros at q=0
            dWgw_qw = np.insert(dWgw_qw, 0, 0, axis=0)
            qqeh = np.insert(qqeh, 0, 0)
            dW_qw = np.insert(dW_qw, 0, 0, axis=0)
            q_cut = q_grid[sort][0] / 2.
        else:
            q_cut = q_grid[sort][1] / 2.

        q0 = np.array([q for q in qqeh if q <= q_cut])
        if len(q0) > 1:  # Integrate arround q=0
            vol = np.pi * (q0[-1] + q0[1] / 2.)**2
            if np.isclose(q0[0], 0):
                weight0 = np.pi * (q0[1] / 2.)**2 / vol
                c = (1 - weight0) / np.sum(q0)
                weights = c * q0
                weights[0] = weight0
            else:
                c = 1 / np.sum(q0)
                weights = c * q0

            dWgw_qw[0] = (np.repeat(weights[:, np.newaxis], len(w_grid),
                                    axis=1) * dW_qw[:len(q0)]).sum(axis=0)

        if not include_q0:  # Omit q=0 contrinution completely.
            dWgw_qw[0] = 0.0

        dWgw_qw = dWgw_qw[isort]  # Put dW back on native grid.
        return dWgw_qw

    def calculate_W_QEH(self, structure, d, layer=0):
        # Module-level _USE_LEGACY_QEH chooses between the modern
        # multi-basis qeh.QEH wrapper (default) and the legacy
        # qeh.old_qeh.Heterostructure path. Both populate self.qqeh,
        # self.wqeh, write <filename>_dW_qw.npz, and return dW_qw.
        if _USE_LEGACY_QEH:
            return self._calculate_W_QEH_legacy(structure, d, layer=layer)
        return self._calculate_W_QEH_modern(structure, d, layer=layer)

    def _calculate_W_QEH_modern(self, structure, d, layer=0):
        from qeh import QEH
        from qeh.heterostructure import expand_layers

        structure = expand_layers(structure)
        self.w_grid = self.omega_w
        wmax = self.w_grid[-1]
        # qmax = (self.q_grid).max()

        # Single layer
        d = np.asarray(d, dtype=float)
        if len(d) == len(structure) - 1:
            d = interlayer_to_thickness(d)
        HS0 = QEH.heterostructure(
            BBfiles=[structure[layer]],
            layerwidth_l=[d[layer] / Bohr],
            wmax=wmax,
            # qmax=qmax / Bohr
        )

        # We only need the V*chi*V part: in dW = W_full - W_iso the bare
        # Coulomb V drops out analytically (same v kernel, same target
        # density basis on both sides), so subtracting it inside QEH
        # avoids a wasted numerical cancellation between two large
        # quantities that share a basis-projection error.
        W0_qw = HS0.get_screened_potential(
            subtract_bare_coulomb=True)[..., 0, 0]

        # Full heterostructure

        HS = QEH.heterostructure(BBfiles=structure, layerwidth_l=d / Bohr,
                                 wmax=wmax,
                                 # qmax=qmax / Bohr
                                 )
        basis_idx = 2 * layer
        W_qw = HS.get_screened_potential(
            subtract_bare_coulomb=True)[..., basis_idx, basis_idx]

        # Difference in screened potential:
        dW_qw = W_qw - W0_qw
        self.wqeh = HS.hs.omega_w
        self.qqeh = HS.hs.q_q

        if self.world.rank == 0:
            data = {'qqeh': self.qqeh,
                    'wqeh': self.wqeh,
                    'dW_qw': dW_qw}
            np.savez(self.filename + "_dW_qw.npz",
                     **data)

        return dW_qw

    def _calculate_W_QEH_legacy(self, structure, d, layer=0):
        """Compute dW_qw using the legacy qeh.old_qeh.Heterostructure.

        The legacy class differs from the modern wrapper in three ways
        that this method bridges:
          - constructor takes (structure, d_interlayer, thicknesses=,
            d0=) instead of (BBfiles=, layerwidth_l=);
          - wmax is in eV, not Hartree;
          - get_screened_potential already returns the layer-projected
            (qN, wN) scalar W_qw (no basis indexing needed).
        Grids are still in Hartree / inverse Bohr on the returned
        object, matching the modern path's contract for self.wqeh /
        self.qqeh.
        """
        from qeh.old_qeh import (
            Heterostructure as LegacyHeterostructure,
            expand_layers as legacy_expand_layers,
        )

        structure = legacy_expand_layers(list(structure))
        self.w_grid = self.omega_w
        # Legacy treats wmax as eV (it does wmax/Hartree internally);
        # self.w_grid is in Hartree.
        wmax_eV = self.w_grid[-1] * Hartree

        d = np.asarray(d, dtype=float)
        N = len(structure)

        # Normalize the input into the (interlayer_distances, thicknesses)
        # pair the legacy class expects. Per-layer thicknesses match the
        # modern path's `layerwidth_l` semantics; interlayer distances
        # follow the center-to-center convention used by legacy `d`.
        if N == 1:
            if len(d) != 1:
                raise ValueError(
                    f'For a single-layer structure, expected d of length '
                    f'1 (the layer thickness in Ang); got length {len(d)}.')
            thicknesses_Ang = np.array([float(d[0])])
            d_interlayer_Ang = np.zeros(0)
        elif len(d) == N - 1:
            d_interlayer_Ang = d
            # Legacy's default thickness rule matches
            # interlayer_to_thickness(d) exactly, so we use the same
            # helper to keep HS0's d0 in sync with the per-layer
            # thicknesses the full HS will derive internally.
            thicknesses_Ang = interlayer_to_thickness(d_interlayer_Ang)
        elif len(d) == N:
            thicknesses_Ang = d
            # Center-to-center distance between adjacent layers.
            d_interlayer_Ang = 0.5 * (thicknesses_Ang[:-1]
                                      + thicknesses_Ang[1:])
        else:
            raise ValueError(
                f'd has length {len(d)}; expected {N - 1} (interlayer '
                f'distances) or {N} (per-layer thicknesses) for a '
                f'{N}-layer structure.')

        # Isolated target layer (W_iso). d0 in the legacy class is the
        # single-layer width when n_layers == 1; passing d=[] keeps the
        # constructor happy (it only sums d for grid sizing).
        HS0 = LegacyHeterostructure(
            structure=[structure[layer]],
            d=np.zeros(0),
            d0=float(thicknesses_Ang[layer]),
            wmax=wmax_eV,
        )
        # subtract_bare_coulomb=True: skip the V add-back so dW = W -
        # W0 doesn't carry the bare-Coulomb cancellation residue
        # between the two basis-projection grids (same rationale as
        # the modern path).
        W0_qw = HS0.get_screened_potential(
            layer=0, subtract_bare_coulomb=True)

        # Full heterostructure (W_full). For N == 1 this is the same
        # object as HS0; we still build it so the rest of the method
        # has a uniform shape (and so HS.frequencies / HS.q_abs come
        # from the full-stack object on multilayer runs).
        if N == 1:
            HS = LegacyHeterostructure(
                structure=list(structure),
                d=np.zeros(0),
                d0=float(thicknesses_Ang[0]),
                wmax=wmax_eV,
            )
        else:
            HS = LegacyHeterostructure(
                structure=list(structure),
                d=d_interlayer_Ang,
                thicknesses=thicknesses_Ang,
                # d0 is only consulted by the substrate branch (which
                # we don't use); legacy still requires it to be a
                # number because of `self.d0 = d0 / Bohr`.
                d0=float(thicknesses_Ang[0]),
                wmax=wmax_eV,
            )
        W_qw = HS.get_screened_potential(
            layer=layer, subtract_bare_coulomb=True)

        dW_qw = W_qw - W0_qw

        # Legacy stores the q/omega grids directly on the instance,
        # already in 1/Bohr and Hartree -- same units the rest of the
        # GWQEH pipeline expects.
        self.wqeh = HS.frequencies
        self.qqeh = HS.q_abs

        if self.world.rank == 0:
            data = {'qqeh': self.qqeh,
                    'wqeh': self.wqeh,
                    'dW_qw': dW_qw}
            np.savez(self.filename + "_dW_qw.npz",
                     **data)

        return dW_qw

    def find_maximum_frequency(self):
        self.epsmin = 10000.0
        self.epsmax = -10000.0
        for kpt in self.gs.kpt_u:
            self.epsmin = min(self.epsmin, kpt.eps_n[0])
            self.epsmax = max(self.epsmax, kpt.eps_n[self.nbands - 1])

        print('Minimum eigenvalue: %10.3f eV' % (self.epsmin * Hartree),
              file=self.fd)
        print('Maximum eigenvalue: %10.3f eV' % (self.epsmax * Hartree),
              file=self.fd)

        return self.epsmax - self.epsmin


def interlayer_to_thickness(d):
    """
    Convert a list/array of inter-plane distances (length N-1)
    to a list/array of layer thicknesses (length N).

    Rule:
        t[0]      = d[0]
        t[i]      = 0.5*(d[i-1] + d[i])   for i = 1 … N-2
        t[N-1]    = d[N-2]

    This guarantees that if every d is the same constant c,
    every t is also c.
    """
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or d.size < 1:
        raise ValueError("`d` must be a 1-D array with at least one element")

    N = d.size + 1
    t = np.empty(N, dtype=float)

    t[0] = d[0]                      # first layer
    if N > 2:                        # interior layers
        t[1:-1] = 0.5 * (d[:-1] + d[1:])
    t[-1] = d[-1]                    # last layer
    return t


class GWmQEHCorrection(GWQEHCorrection):
    """GW self-energy correction using the mQEH (multi-basis QEH) method.

    Generalizes the monopole G-Delta-W approach to the full mQEH basis by:
    1. Including multiple in-plane G_parallel vectors (not just G=0)
    2. Using all mQEH density/potential basis functions per layer
    3. Projecting pair densities onto the mQEH basis via expansion
       coefficients C^{n,l0}_{m,alpha}(G_parallel)

    The self-energy correction is (Eq. 15 of the theory document):

        Delta-Sigma_{nk} = (i/2pi) sum_{mq} int dw'
            1/(w - w' - eps_{m,k-q})
            * (1/A) sum_{G_par} sum_{alpha,beta}
                [C^{n,l0}_{m,alpha}(G_par)]* C^{n,l0}_{m,beta}(G_par)
                * Delta-W_{l0 alpha, l0 beta}(|q + G_par|, w')

    Parameters
    ----------
    calc : str or PAW object
        GPAW calculator object or filename of saved calculator object.
    ecut_mqeh : float
        Plane-wave cutoff (in eV) for the in-plane G_parallel vectors.
        Controls how many G_parallel shells are included beyond G=0.
        Default: 50 eV.
    All other parameters are the same as GWQEHCorrection.
    """

    def __init__(self, calc, gwfile=None, filename=None, kpts=[0], bands=None,
                 structure=None, d=None, layer=0,
                 qqeh=None, wqeh=None,
                 txt=sys.stdout, world=mpi.world, domega0=0.025,
                 omega2=10.0, eta=0.1, include_q0=True, metal=False,
                 restart=False, ecut_mqeh=50.0,
                 dW_qw_matrix=None, drho_qzi=None,
                 z_z_qeh=None, dz_qeh=None):

        self.ecut_mqeh = ecut_mqeh / Hartree

        if metal:
            # _interpolate_mqeh_data does not implement the parent's
            # metal=True semantics (strip q=0 from the QEH interpolation
            # grid; shift the q_cut accordingly). Refuse rather than
            # silently treating a metal as a semiconductor.
            raise NotImplementedError(
                'metal=True is not yet supported for GWmQEHCorrection; '
                'the matrix q-interpolation in _interpolate_mqeh_data '
                'would silently fall back to the non-metal q_cut.')

        # Used by _interpolate_mqeh_data to decide whether to zero or
        # weight-average the small-q ring of the dW matrix.
        self._include_q0 = include_q0

        # State set by _setup_dW (override below). NOTE: this class
        # deliberately does NOT carry the QEH potential basis
        # (phi_qiz). The mQEH self-energy is bilinear in the rho-LS
        # coefficients of the pair density (see _calculate_sigma_mqeh),
        # so the truncated phi basis from Layer.get_phi_qaz must never
        # appear here. If you find yourself reaching for phi, that is a
        # sign that the projection has reverted to the old (buggy)
        # phi-inner-product form.
        self.dW_qw_matrix = None
        self.drho_qzi_target = None
        self.nbasis = None
        self.layer_index = layer
        self.qqeh_matrix = None
        self.wqeh_matrix = None
        self.z_z_qeh = None
        self.dz_qeh = None

        # Stash user-provided mQEH data so the _setup_dW override
        # (dispatched from super().__init__()) can pick it up.
        self._init_dW_qw_matrix = dW_qw_matrix
        self._init_drho_qzi = drho_qzi
        self._init_z_z_qeh = z_z_qeh
        self._init_dz_qeh = dz_qeh

        # The parent's `dW_qw` scalar pipeline is not used by mQEH; we
        # never pass a scalar here.
        super().__init__(
            calc=calc, gwfile=gwfile, filename=filename, kpts=kpts,
            bands=bands, structure=structure, d=d, layer=layer,
            dW_qw=None, qqeh=qqeh, wqeh=wqeh,
            txt=txt, world=world, domega0=domega0,
            omega2=omega2, eta=eta, include_q0=include_q0,
            metal=metal, restart=restart)

        # mQEH must work in a coordinate frame where the third lattice
        # vector is the out-of-plane direction (Cartesian z). The G_par
        # grouping and Gz extraction rely on this.
        cell_cv = self.gs.gd.cell_cv
        assert np.allclose(cell_cv[2, :2], 0) and \
            np.allclose(cell_cv[:2, 2], 0), \
            ('GWmQEHCorrection assumes the third lattice vector is along '
             'Cartesian z and orthogonal to the in-plane axes.')

    def _setup_dW(self, *, dW_qw, qqeh, wqeh, structure, d, layer,
                  restart, include_q0, metal):
        """Install the mQEH dW matrix; the parent's scalar pipeline is
        not used.

        ``self.dW_qw`` is intentionally never set: nothing in the mQEH
        path reads it (the per-q evaluation goes through
        ``self._eval_dW_on_gwgrid`` on the precomputed matrix splines).
        ``include_q0`` is honored via ``self._include_q0`` (stored in
        ``__init__``) inside ``_interpolate_mqeh_data``. ``metal`` is
        rejected at construction time, so it cannot reach this method
        as ``True``. ``dW_qw`` is ignored.
        """
        del dW_qw, include_q0, metal       # not used by mQEH

        # Path 1: user-supplied matrix (synthetic-data / test path).
        if self._init_dW_qw_matrix is not None:
            assert qqeh is not None and wqeh is not None, \
                ('GWmQEHCorrection: qqeh and wqeh must be supplied '
                 'when dW_qw_matrix is provided directly')
            self.qqeh = np.asarray(qqeh)
            self.wqeh = np.asarray(wqeh)
            self._install_mqeh_matrix(
                dW_qw_matrix=self._init_dW_qw_matrix,
                drho_qzi=self._init_drho_qzi,
                z_z_qeh=self._init_z_z_qeh,
                dz_qeh=self._init_dz_qeh,
                qqeh_matrix=self.qqeh,
                wqeh_matrix=self.wqeh)
            return

        # Path 2: restart from <filename>_dW_qw.npz.
        if restart:
            try:
                data = np.load(self.filename + '_dW_qw.npz')
            except IOError:
                pass
            else:
                required = ('dW_qw_matrix', 'drho_qzi', 'z_z_qeh',
                            'dz_qeh', 'nbasis', 'qqeh', 'wqeh')
                if all(k in data.files for k in required):
                    self.qqeh = data['qqeh']
                    self.wqeh = data['wqeh']
                    self._install_mqeh_matrix(
                        dW_qw_matrix=data['dW_qw_matrix'],
                        drho_qzi=data['drho_qzi'],
                        z_z_qeh=data['z_z_qeh'],
                        dz_qeh=float(data['dz_qeh']),
                        qqeh_matrix=self.qqeh,
                        wqeh_matrix=self.wqeh)
                    print('mQEH matrix data loaded from file',
                          file=self.fd)
                    return

        # Path 3: fresh QEH computation (sets state on self).
        self.calculate_W_QEH(structure, d, layer)

    def _install_mqeh_matrix(self, *, dW_qw_matrix, drho_qzi, z_z_qeh,
                             dz_qeh, qqeh_matrix, wqeh_matrix):
        """Set mQEH matrix state and build the q / omega interpolators.

        Single chokepoint for state installation so all three paths
        (synthetic data, restart, fresh QEH) go through the same code.
        """
        dW = np.asarray(dW_qw_matrix)
        self.dW_qw_matrix = dW
        self.nbasis = int(dW.shape[2])
        self.drho_qzi_target = np.asarray(drho_qzi)
        self.z_z_qeh = np.asarray(z_z_qeh)
        self.dz_qeh = float(dz_qeh)
        self.qqeh_matrix = np.asarray(qqeh_matrix).copy()
        self.wqeh_matrix = np.asarray(wqeh_matrix).copy()
        # Defend against silent unit drift in the restart / synthetic
        # paths.
        if len(self.z_z_qeh) > 1:
            assert np.isclose(self.dz_qeh,
                              self.z_z_qeh[1] - self.z_z_qeh[0]), \
                'dz_qeh inconsistent with z_z_qeh spacing'
        # Compute the GW q-magnitude grid that _interpolate_mqeh_data's
        # q -> 0 averaging block depends on. Before the legacy-scalar
        # refactor this attribute was set as a side effect of the
        # parent's get_W_on_grid; with that path gone we have to
        # populate it directly here, otherwise _q0_dW_Wab stays None
        # and include_q0 / small-q averaging silently no-op.
        rcell_cv = 2 * pi * np.linalg.inv(self.gs.gd.cell_cv).T
        q_vs = np.dot(self.qd.ibzk_kc, rcell_cv)
        self.q_grid = (q_vs**2).sum(axis=1) ** 0.5
        self._interpolate_mqeh_data()

    def calculate_W_QEH(self, structure, d, layer=0):
        """Compute and install the full mQEH Delta-W matrix.

        State-setter: writes ``self.dW_qw_matrix`` and friends via
        ``_install_mqeh_matrix``. Does not return anything.
        """
        # MQEH (not QEH) is the right driver here. The self-energy
        # bilinear in _calculate_sigma_mqeh is d^dagger W d where d
        # are rho-LS coefficients of the pair density, which is only
        # correct when W = <rho|W|rho>. That is exactly what MQEH's
        # get_screened_potential returns (its Coulomb-kernel override
        # is V_{ij} = <rho_i|dphi_j>; see qeh/mqeh.py:13-31).
        # QEH.get_screened_potential instead returns
        # V_{ij} = (gphi @ phi @ dphi) which is in the dual-phi basis;
        # combined with the rho-LS projection here, dimensionful basis
        # factors (g_phi ~ 1/Lz from phi=1, S^{-1} ~ 1/<rho|rho>) end
        # up multiplying sigma by ~Lz^2 / (basis overlap)^2, giving
        # the wild O(100 eV) overestimate observed for mbb-format
        # BBs whose rho is biorthogonal to phi (not L2-normalized).
        from qeh import MQEH
        from qeh.heterostructure import expand_layers

        structure = expand_layers(structure)
        self.w_grid = self.omega_w
        wmax = self.w_grid[-1]

        d = np.asarray(d, dtype=float)
        if len(d) == len(structure) - 1:
            d = interlayer_to_thickness(d)

        # Single layer (isolated monolayer)
        HS0 = MQEH.heterostructure(
            BBfiles=[structure[layer]],
            layerwidth_l=[d[layer] / Bohr],
            wmax=wmax,
        )

        # Drop the bare-Coulomb part of W on both sides of the
        # difference: V cancels exactly in dW = W_full - W_iso, so we
        # only need V*chi*V here. See parent calculate_W_QEH for the
        # full argument.
        W0_qwij = HS0.get_screened_potential(subtract_bare_coulomb=True)

        # Full heterostructure
        HS = MQEH.heterostructure(
            BBfiles=structure,
            layerwidth_l=d / Bohr,
            wmax=wmax,
        )

        W_qwij = HS.get_screened_potential(subtract_bare_coulomb=True)

        # Number of basis functions for the target layer
        nbasis_target = HS.hs.layers_l[layer].bb.aN

        # Extract the block of W corresponding to the target layer
        i0 = sum(HS.hs.layers_l[l].bb.aN for l in range(layer))
        i1 = i0 + nbasis_target

        # Delta-W for the target layer block
        # Shape: (nq, nw, nbasis, nbasis)
        dW_qwab = (W_qwij[:, :, i0:i1, i0:i1]
                   - W0_qwij[:, :, :nbasis_target, :nbasis_target])

        # Density basis functions for the target layer on the het z-grid.
        # We intentionally do NOT take the potential basis: the mQEH
        # self-energy is a rho-bilinear (see _calculate_sigma_mqeh),
        # and Layer.get_phi_qaz hard-zeros phi outside the layer width,
        # which would break the projection.
        target_layer = HS.hs.layers_l[layer]
        drho_qzi = np.array(
            [target_layer.get_drho_qza(iq_q=[iq])[0]
             for iq in range(HS.hs.qN)])

        qqeh = HS.hs.q_q.copy()
        wqeh = HS.hs.omega_w.copy()
        # Set self.qqeh / self.wqeh for npz consumers; matrix copies
        # are installed by _install_mqeh_matrix below.
        self.qqeh = qqeh
        self.wqeh = wqeh

        self._install_mqeh_matrix(
            dW_qw_matrix=dW_qwab,
            drho_qzi=drho_qzi,
            z_z_qeh=HS.hs.z_z.copy(),
            dz_qeh=HS.hs.dz,
            qqeh_matrix=qqeh,
            wqeh_matrix=wqeh)

        # Save for restart. No 'dW_qw' scalar -- mQEH does not use it,
        # and writing the (0,0) slice was a misnomer (it is the first
        # eigenmode of V*chi, not the W&T-2017 monopole).
        if self.world.rank == 0:
            data = {'qqeh': qqeh,
                    'wqeh': wqeh,
                    'dW_qw_matrix': dW_qwab,
                    'drho_qzi': drho_qzi,
                    'z_z_qeh': self.z_z_qeh,
                    'dz_qeh': self.dz_qeh,
                    'nbasis': self.nbasis}
            np.savez(self.filename + '_dW_qw.npz', **data)

    def _interpolate_mqeh_data(self):
        """Pre-compute interpolators for the mQEH Delta-W matrix and
        basis functions so they can be evaluated at arbitrary |q+G_par|.

        Delta-W is also pre-interpolated along the frequency axis from
        the QEH omega-grid onto the GW omega-grid, so per-call queries
        return arrays already on the GW grid (no per-call CubicSpline
        construction in omega).
        """
        from scipy.interpolate import CubicSpline

        qqeh = self.qqeh_matrix
        sortq = np.argsort(qqeh)
        self._qqeh_sorted = qqeh[sortq]

        # Sort along q for all interpolated arrays
        dW_sorted = self.dW_qw_matrix[sortq]          # (nq, nw_qeh, nb, nb)
        drho_sorted = self.drho_qzi_target[sortq]     # (nq, nz, nb)
        nb = self.nbasis
        nw_qeh = dW_sorted.shape[1]
        nz = drho_sorted.shape[1]

        # Pre-interpolate Delta-W along the omega axis from the QEH
        # frequency grid to the GW frequency grid. This makes per-call
        # queries a single q-evaluation that already returns shape
        # (nw_gw, nb, nb), eliminating the per-Gpar per-iq construction
        # of CubicSpline objects in omega.
        wqeh = self.wqeh_matrix
        w_spl_re = CubicSpline(wqeh, dW_sorted.real, axis=1,
                               extrapolate=True)
        w_spl_im = CubicSpline(wqeh, dW_sorted.imag, axis=1,
                               extrapolate=True)
        dW_sorted_W = (w_spl_re(self.omega_w)
                       + 1j * w_spl_im(self.omega_w))   # (nq, nw_gw, nb, nb)

        # Single multi-output splines in q for Delta-W and drho.
        # Real and imag parts split so we can use CubicSpline (which
        # only takes real data). NB: no phi spline -- mQEH does not
        # access the truncated potential basis.
        self._dW_spline_re = CubicSpline(
            self._qqeh_sorted, dW_sorted_W.real, axis=0, extrapolate=True)
        self._dW_spline_im = CubicSpline(
            self._qqeh_sorted, dW_sorted_W.imag, axis=0, extrapolate=True)
        self._drho_spline_re = CubicSpline(
            self._qqeh_sorted, drho_sorted.real, axis=0, extrapolate=True)
        self._drho_spline_im = CubicSpline(
            self._qqeh_sorted, drho_sorted.imag, axis=0, extrapolate=True)

        self._nw_qeh = nw_qeh
        self._nz_qeh = nz
        self._qqeh_max = float(self._qqeh_sorted[-1])

        # q -> 0 averaging for the matrix (mirrors the parent's
        # treatment of dWgw_qw[0]). The mQEH Delta-W matrix can diverge
        # as q -> 0 (Coulomb-like long-range part), so when |q+G_par|
        # falls below q_cut we substitute a weighted average over the
        # smallest q-points instead of the raw spline extrapolation.
        # _q0_dW_Wab is on the GW omega-grid (the only one we evaluate).
        self._q0_dW_Wab = None
        self._q0_cut = 0.0
        q_grid = getattr(self, 'q_grid', None)
        if q_grid is not None and len(q_grid) > 1:
            q_sorted = np.sort(q_grid)
            # Match the parent's q_cut: half the first nonzero |q| in the
            # GW grid. For non-metals q_sorted[0] is the Gamma point (0)
            # and we want q_sorted[1]; for metals the parent strips q=0
            # and uses [0] of the truncated array, which is the same
            # nonzero value. Using "first nonzero" works in both cases
            # without an unsafe q_sorted[0]/2 = 0 cut for metals.
            nonzero = q_sorted[~np.isclose(q_sorted, 0)]
            if len(nonzero) == 0:
                return
            self._q0_cut = nonzero[0] / 2.0
            q0 = self._qqeh_sorted[self._qqeh_sorted <= self._q0_cut]
            nw_gw = len(self.omega_w)
            if not self._include_q0:
                # Match the parent's behavior: zero out the matrix
                # for |q+G_par| < q0_cut.
                self._q0_dW_Wab = np.zeros((nw_gw, nb, nb), dtype=complex)
            elif len(q0) > 1:
                # Weighted average over the small-q ring, weights ~ q
                # (replicating the area-element of the q -> 0 disk).
                if np.isclose(q0[0], 0):
                    vol = np.pi * (q0[-1] + q0[1] / 2.0)**2
                    weight0 = np.pi * (q0[1] / 2.0)**2 / vol
                    c = (1 - weight0) / np.sum(q0)
                    weights = c * q0
                    weights[0] = weight0
                else:
                    c = 1.0 / np.sum(q0)
                    weights = c * q0
                # dW_sorted_W has the same q ordering as _qqeh_sorted
                # and is already on the GW omega-grid.
                small_dW = dW_sorted_W[:len(q0)]   # (n_small, nw_gw, nb, nb)
                self._q0_dW_Wab = np.tensordot(
                    weights, small_dW, axes=(0, 0))  # (nw_gw, nb, nb)

    def _eval_dW_on_gwgrid(self, q_abs):
        """Evaluate Delta-W matrix at |q| on the GW frequency grid.

        Returns array of shape (nw_gw, nbasis, nbasis).  Substitutes
        the small-q average when q_abs <= _q0_cut.
        """
        if (self._q0_dW_Wab is not None
                and q_abs <= self._q0_cut):
            return self._q0_dW_Wab.copy()
        if q_abs > self._qqeh_max:
            if not getattr(self, '_warned_qmax', False):
                print(('WARNING: evaluating Delta-W at |q+G_par|=%.3f '
                       'Bohr^-1 > qqeh.max()=%.3f; results rely on '
                       'spline extrapolation. Consider increasing the '
                       'mQEH q_max or decreasing ecut_mqeh.')
                      % (q_abs, self._qqeh_max), file=self.fd)
                self._warned_qmax = True
        return (self._dW_spline_re(q_abs)
                + 1j * self._dW_spline_im(q_abs))

    def _eval_drho(self, q_abs):
        """Evaluate density basis functions at |q|.

        Returns array of shape (nz, nbasis).
        """
        return (self._drho_spline_re(q_abs)
                + 1j * self._drho_spline_im(q_abs))

    def calculate_QEH(self):
        """Calculate the mQEH self-energy contribution.

        This overrides the parent method to include all G_parallel vectors
        and the full mQEH basis expansion.
        """
        print('Calculating mQEH self-energy contribution', file=self.fd)

        kd = self.gs.kd
        atomrotations = self.gs.atomrotations

        # Reset
        self.sigma_sin = np.zeros(self.shape)
        self.dsigma_sin = np.zeros(self.shape)

        # Get KS eigenvalues and occupation numbers
        b1, b2 = self.bands
        for i, k in enumerate(self.kpts):
            for s in range(self.nspins):
                kpt = self.gs.kpt_ks[k][s]
                self.eps_sin[s, i] = kpt.eps_n[b1:b2]
                self.f_sin[s, i] = kpt.f_n[b1:b2] / kpt.weight

        mykpts = [self.kptpair_factory.get_k_point(s, K, n1, n2)
                  for s, K, n1, n2 in self.mysKn1n2]

        # Reciprocal cell for computing |q + G_par|
        rcell_cv = 2 * pi * np.linalg.inv(self.gs.gd.cell_cv).T
        L = abs(self.gs.gd.cell_cv[2, 2])
        A = abs(np.linalg.det(self.gs.gd.cell_cv[:2, :2]))
        N_c = self.gs.gd.N_c

        Nq = len(self.qd.ibzk_kc)
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            self.nq = iq
            self.save_state_file()

            qcstr = '(' + ', '.join(['%.3f' % x for x in q_c]) + ')'
            print('Calculating mQEH contribution from IBZ q-point '
                  '#%d/%d q_c=%s' % (iq, Nq, qcstr), file=self.fd)

            q_v = np.dot(q_c, rcell_cv)

            # Setup q-point descriptor with HIGHER ecut for G_parallel
            pd0 = SingleQPWDescriptor.from_q(
                q_c, self.ecut_mqeh, self.gs.gd, gammacentered=True)
            G_Gv = pd0.get_reciprocal_vectors()       # G + q vectors
            G0_Gv = pd0.get_reciprocal_vectors(add_q=False)  # G vectors only
            nG = len(G_Gv)

            # Group G-vectors by in-plane component G_parallel = (Gx, Gy)
            # For each unique G_parallel, collect the G_z indices
            Gpar_Gv = G0_Gv[:, :2]  # In-plane G components
            Gz_G = G0_Gv[:, 2]      # Out-of-plane G components

            # Find unique G_parallel vectors
            # Round to avoid floating point issues
            Gpar_rounded = np.round(Gpar_Gv, decimals=8)
            unique_Gpar, inverse_idx = np.unique(
                Gpar_rounded, axis=0, return_inverse=True)
            n_Gpar = len(unique_Gpar)

            print('  Number of G-vectors: %d, unique G_parallel: %d'
                  % (nG, n_Gpar), file=self.fd)

            # For each unique G_parallel, compute |q + G_parallel|
            qplusGpar_abs = np.zeros(n_Gpar)
            for ig, Gpar_v in enumerate(unique_Gpar):
                qpG_v = q_v[:2] + Gpar_v
                qplusGpar_abs[ig] = np.sqrt(np.sum(qpG_v**2))

            # Evaluate mQEH Delta-W at each |q + G_parallel|
            # and prepare Hilbert-transformed W matrices
            nw = self.nw

            # For each G_parallel, we need the Hilbert-transformed
            # Delta-W matrix in the basis function space
            # Shape: (n_Gpar, 2*nw, nbasis, nbasis)
            nb = self.nbasis
            Wpm_Gpar = np.zeros(
                (n_Gpar, 2 * nw, nb, nb), dtype=complex)

            for ig in range(n_Gpar):
                q_abs = qplusGpar_abs[ig]
                # Evaluate Delta-W at this |q+G_par| directly on the
                # GW frequency grid; the omega-direction interpolation
                # was precomputed in _interpolate_mqeh_data.
                dW_wab = self._eval_dW_on_gwgrid(q_abs)
                # dW is Hartree*Bohr^2; *= L makes x*L = 1/(N_q*2pi*A).
                dW_wab *= L

                # Set up Wpm for Hilbert transform
                Wpm_Gpar[ig, :nw] = dW_wab
                Wpm_Gpar[ig, nw:] = dW_wab.copy()

                # Apply Hilbert transforms
                self.htp(Wpm_Gpar[ig, :nw])
                self.htm(Wpm_Gpar[ig, nw:])

            # Density basis functions rho_alpha(|q+G_par|; z) on the QEH
            # z-grid, and the rho overlap S_{ab}(|q+G_par|) on the same
            # grid. These are the only basis-side ingredients of the mQEH
            # sigma; phi never appears.
            drho_Gpar_za = np.zeros(
                (n_Gpar, self._nz_qeh, nb), dtype=complex)
            S_Gpar_ab = np.zeros((n_Gpar, nb, nb), dtype=complex)
            for ig in range(n_Gpar):
                drho_za = self._eval_drho(qplusGpar_abs[ig])
                drho_Gpar_za[ig] = drho_za
                # S_{ab} = int rho_a*(z) rho_b(z) dz on the het z-grid.
                # Generally != delta_ab because the het-grid integration
                # is not the BB-grid integration used for biorthogonality.
                S_Gpar_ab[ig] = (drho_za.conj().T @ drho_za) * self.dz_qeh

            # Precompute per-(G_par-group) z-Fourier phase factors and
            # G-vector index lists. These depend only on iq (and the FFT
            # geometry), not on symmetry / kpt / band, so we build them
            # once per iq and reuse inside _calculate_sigma_mqeh.
            Q_G = pd0.Q_qG[0]
            i_cG = np.array(np.unravel_index(Q_G, N_c))
            Gz_idx = i_cG[2]
            Gz_idx_wrapped = np.where(Gz_idx > N_c[2] // 2,
                                      Gz_idx - N_c[2], Gz_idx)
            z_qeh = self.z_z_qeh
            twopi_over_Lz = 2 * pi / L
            G_indices_per_Gpar = []
            phase_zg_per_Gpar = []
            for ig in range(n_Gpar):
                idx = np.where(inverse_idx == ig)[0]
                G_indices_per_Gpar.append(idx)
                if len(idx) == 0:
                    phase_zg_per_Gpar.append(None)
                    continue
                Gz_values = Gz_idx_wrapped[idx] * twopi_over_Lz
                phase_zg_per_Gpar.append(
                    np.exp(1j * np.outer(z_qeh, Gz_values)))

            # PAW corrections
            self.Q_aGii = self.gs.pair_density_paw_corrections(pd0).Q_aGii

            # Loop over symmetry-related q-points
            Q1 = self.qd.ibz2bz_k[iq]
            Q2s = set()
            for s, Q2 in enumerate(self.qd.bz2bz_ks[Q1]):
                if Q2 >= 0 and Q2 not in Q2s:
                    Q2s.add(Q2)

            for Q2 in Q2s:
                s = self.qd.sym_k[Q2]
                self.s = s
                U_cc = self.qd.symmetry.op_scc[s]
                time_reversal = self.qd.time_reversal_k[Q2]
                self.sign = 1 - 2 * time_reversal
                Q_c = self.qd.bzk_kc[Q2]
                d_c = self.sign * np.dot(U_cc, q_c) - Q_c
                assert np.allclose(d_c.round(), d_c)

                for u1, kpt1 in enumerate(mykpts):
                    K2 = kd.find_k_plus_q(Q_c, [kpt1.K])[0]
                    kpt2 = self.kptpair_factory.get_k_point(
                        kpt1.s, K2, 0, self.nbands,
                        blockcomm=self.blockcomm)
                    k1 = kd.bz2ibz_k[kpt1.K]
                    i = self.kpts.index(k1)

                    def coordinate_transformation(q_c):
                        return self.sign * np.dot(U_cc, q_c)

                    I_G = phase_shifted_fft_indices(
                        kpt1.k_c, kpt2.k_c, pd0,
                        coordinate_transformation=coordinate_transformation)

                    pos_av = self.gs.get_pos_av()
                    M_vv = np.dot(
                        self.gs.gd.cell_cv.T,
                        np.dot(U_cc.T,
                               np.linalg.inv(self.gs.gd.cell_cv).T))
                    Q_aGii = []
                    for a, Q_Gii in enumerate(self.Q_aGii):
                        x_G = np.exp(1j * np.dot(
                            G_Gv,
                            (pos_av[a] - np.dot(M_vv, pos_av[a]))))
                        R_sii = atomrotations.get_R_asii()[a]
                        U_ii = R_sii[self.s]
                        Q_Gii = np.dot(
                            np.dot(U_ii,
                                   Q_Gii * x_G[:, None, None]),
                            U_ii.T).transpose(1, 0, 2)
                        if self.sign == -1:
                            Q_Gii = Q_Gii.conj()
                        Q_aGii.append(Q_Gii)

                    for n in range(kpt1.n2 - kpt1.n1):
                        ut1cc_R = kpt1.ut_nR[n].conj()
                        eps1 = kpt1.eps_n[n]
                        C1_aGi = [
                            np.dot(Qa_Gii, P1_ni[n].conj())
                            for Qa_Gii, P1_ni in zip(Q_aGii,
                                                     kpt1.P_ani)]

                        # Calculate pair densities for ALL G-vectors
                        n_mG = self.pair_calc.calculate_pair_density(
                            ut1cc_R, C1_aGi, kpt2, pd0, I_G)

                        if self.sign == 1:
                            n_mG = n_mG.conj()

                        # Compute expansion coefficients and self-energy
                        f_m = kpt2.f_n
                        deps_m = eps1 - kpt2.eps_n

                        sigma, dsigma = self._calculate_sigma_mqeh(
                            n_mG, deps_m, f_m, Wpm_Gpar,
                            G_indices_per_Gpar, phase_zg_per_Gpar,
                            drho_Gpar_za, S_Gpar_ab, L)

                        nn = kpt1.n1 + n - self.bands[0]
                        self.sigma_sin[kpt1.s, i, nn] += sigma
                        self.dsigma_sin[kpt1.s, i, nn] += dsigma

        self.world.sum(self.sigma_sin)
        self.world.sum(self.dsigma_sin)

        self.complete = True
        self.save_state_file()

        return self.sigma_sin, self.dsigma_sin

    def _calculate_sigma_mqeh(self, n_mG, deps_m, f_m, Wpm_Gpar,
                              G_indices_per_Gpar, phase_zg_per_Gpar,
                              drho_Gpar_za, S_Gpar_ab, Lz):
        """Calculate self-energy contribution in the mQEH basis.

        For each G_parallel, fit the pair density onto the mQEH density
        basis by least squares (the rho basis is in general not
        orthogonal on the heterostructure z-grid), then contract the
        resulting rho-coefficients with the Delta-W matrix.

        In the mQEH convention (qeh.MQEH), W_qwij = V + V chi V is the
        symmetric <rho|W|rho> kernel: it takes rho-coefficients of a
        probe density in and returns rho-projections of the screened
        potential out (see qeh/mqeh.py:13-31, 228-269). The self-energy
        bilinear is therefore

            Sigma ~ d^dagger W^mQEH d

        with d_alpha the least-squares rho-coefficients of bar_rho,
        obtained from the normal equation

            S_{ab} d_b = R_a,
            R_a = int rho_a*(z) bar_rho(z) dz,
            S_{ab} = int rho_a*(z) rho_b(z) dz.

        The truncated potential basis phi_qiz does not appear -- using
        it would silently revert to the old phi-inner-product form,
        which is only equivalent under exact biorthogonality on the het
        z-grid (broken by Layer.get_phi_qaz's hard zeroing outside the
        layer width).

        Parameters
        ----------
        n_mG : ndarray (nbands, nG)
            Pair densities in plane-wave basis for all G-vectors.
        deps_m : ndarray (nbands,)
            Energy differences.
        f_m : ndarray (nbands,)
            Occupation numbers.
        Wpm_Gpar : ndarray (n_Gpar, 2*nw, nbasis, nbasis)
            Hilbert-transformed Delta-W matrices for each G_parallel.
        G_indices_per_Gpar : list of ndarray
            For each unique G_parallel group, the indices into n_mG's
            G-axis that belong to it.
        phase_zg_per_Gpar : list of ndarray or None
            For each unique G_parallel group, exp(i Gz z_qeh) on the
            QEH z-grid with shape (nz_qeh, n_gz). None for empty groups.
        drho_Gpar_za : ndarray (n_Gpar, nz_qeh, nbasis)
            Density basis functions rho_alpha(|q+G_par|; z).
        S_Gpar_ab : ndarray (n_Gpar, nbasis, nbasis)
            Rho overlap matrix S_{ab} = int rho_a* rho_b dz on the het
            z-grid.
        Lz : float
            DFT cell height (for inverse-FFT normalization).
        """
        o_m = abs(deps_m)
        sgn_m = np.sign(deps_m + 1e-15)
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2

        nw = len(self.omega_w)
        nb = self.nbasis
        n_Gpar = Wpm_Gpar.shape[0]

        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        o1_m = self.omega_w[w_m]
        o2_m = self.omega_w[w_m + 1]
        x = 1.0 / (self.qd.nbzkpts * 2 * pi * self.vol)

        # d_{m, ig, a} = (S^{-1} R)_{m, ig, a} are the rho-LS coefficients
        # of the pair density bar_rho^{nm}(G_par; z) in the rho basis.
        # rho(G_par, z) = sum_{G_z} n(G_par, G_z) e^{i G_z z} / Lz
        # (inverse z-FFT consistent with GPAW's e^{-iG.r} forward sign).
        d_mGpar_a = np.zeros((len(deps_m), n_Gpar, nb), dtype=complex)

        for ig in range(n_Gpar):
            G_indices = G_indices_per_Gpar[ig]
            if len(G_indices) == 0:
                continue
            phase_zg = phase_zg_per_Gpar[ig]      # (nz_qeh, n_gz)
            n_m_gz = n_mG[:, G_indices]           # (nbands, n_gz)
            rho_mz = n_m_gz @ phase_zg.T / Lz     # (nbands, nz_qeh)
            drho_za = drho_Gpar_za[ig]            # (nz_qeh, nbasis)
            # R_{m, a} = int rho_a*(z) bar_rho^{nm}(z) dz
            R_m_a = (rho_mz @ drho_za.conj()) * self.dz_qeh
            # Normal equation: S @ d.T = R.T  =>  d = solve(S, R.T).T
            d_mGpar_a[:, ig, :] = np.linalg.solve(
                S_Gpar_ab[ig], R_m_a.T).T

        # Prefactor: x = 1/(N_q*2pi*Omega) and dW carries an extra L,
        # so x*L = 1/(N_q*2pi*A) matches Eq.(9) of W&T 2017.
        sigma = 0.0
        dsigma = 0.0

        for o, o1, o2, sgn, s, w, d_Gpar_a in zip(
                o_m, o1_m, o2_m, sgn_m, s_m, w_m, d_mGpar_a):
            p = x * sgn
            sigma1 = 0.0
            sigma2 = 0.0

            for ig in range(n_Gpar):
                d_a = d_Gpar_a[ig]                    # (nbasis,)
                W1_ab = Wpm_Gpar[ig, s * nw + w]      # (nbasis, nbasis)
                W2_ab = Wpm_Gpar[ig, s * nw + w + 1]  # (nbasis, nbasis)

                sigma1 += p * (d_a.conj() @ W1_ab @ d_a).imag
                sigma2 += p * (d_a.conj() @ W2_ab @ d_a).imag

            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)

        return sigma, dsigma
