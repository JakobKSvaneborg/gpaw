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
                 z_z_qeh=None, dz_qeh=None,
                 dump_pair_densities=False,
                 dump_pair_densities_max_samples=40,
                 lagrange_constrained_ls=True,
                 chi_files=None,
                 build_bb_on_query_grid=False,
                 bb_aN=4, bb_zN=200, bb_output_dir=None):

        self.ecut_mqeh = ecut_mqeh / Hartree

        # Enforce charge conservation in the rho-LS projection by
        # imposing the single linear constraint
        #     sum_a d_a * (int rho_a(z) dz) = int bar_rho(z) dz
        # via a Lagrange multiplier. The in-plane pair density
        # bar_rho^{nm}(q, z) integrates to zero in z as q -> 0 (charge
        # conservation), and that cancellation is precisely what kills
        # the 2pi/q Coulomb divergence in the bilinear. An
        # unconstrained least-squares fit on a truncated rho basis can
        # leave a residual with non-zero integral, which then couples
        # to the divergent small-q W and produces nonsense. Default to
        # True; set False to A/B test against the unconstrained LS.
        self._lagrange_constrained_ls = bool(lagrange_constrained_ls)

        # Exact-grid path: build mqeh building blocks from raw chi files
        # on the precise list of (|q+G_par|, omega) values that the GW
        # path will query, eliminating spline interpolation in q and
        # omega entirely. `chi_files` is a per-layer list of *-chi.npz
        # paths (with repetition for repeated materials). If
        # `build_bb_on_query_grid` is True, `_setup_dW` ignores
        # `structure` and instead dispatches to
        # `_build_and_install_mqeh_from_chi`, which (i) gathers the W
        # query grid from `self.qd.ibzk_kc` and `self.ecut_mqeh`,
        # (ii) calls `qeh.bb_calculator.bb_builder.interpolate_chi_to_bb`
        # on each unique chi file with that grid, and (iii) loads the
        # resulting mbb files via `qeh.MQEH.heterostructure`.
        self._chi_files = (list(chi_files) if chi_files is not None
                           else None)
        self._build_bb_on_query_grid = bool(build_bb_on_query_grid)
        self._bb_aN = int(bb_aN)
        self._bb_zN = int(bb_zN)
        self._bb_output_dir = (str(bb_output_dir)
                               if bb_output_dir is not None else None)
        if self._build_bb_on_query_grid and not self._chi_files:
            raise ValueError(
                'GWmQEHCorrection: chi_files must be a non-empty list '
                'when build_bb_on_query_grid=True')

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

        # When dump_pair_densities is True, calculate_QEH saves a small
        # ``<filename>_mqeh_pair_densities.npz`` containing the first
        # ``dump_pair_densities_max_samples`` (rho_mz, drho_za, d) tuples
        # encountered, plus their (iq, ig, m, |q+G_par|) metadata. The
        # companion script ``gpaw/response/gwmqeh_plot_pair_densities.py``
        # loads this file and plots each pair density alongside its
        # rho-LS fit ``sum_a d_a rho_a(z)`` so the alignment between the
        # DFT pair density (in the qeh frame, after the z-frame shift)
        # and the mQEH basis can be eyeballed cheaply on a laptop.
        self._dump_pair_densities = bool(dump_pair_densities)
        self._dump_max = int(dump_pair_densities_max_samples)
        self._pair_density_samples = []

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

        # Path 4: build mbb files from chi files on the exact W query
        # grid, then install. Avoids spline interpolation entirely.
        if self._build_bb_on_query_grid:
            self._build_and_install_mqeh_from_chi(d=d, layer=layer)
            return

        # Path 3: fresh QEH computation from pre-built BBs (spline path).
        self.calculate_W_QEH(structure, d, layer)

    def _install_mqeh_matrix(self, *, dW_qw_matrix, drho_qzi, z_z_qeh,
                             dz_qeh, qqeh_matrix, wqeh_matrix,
                             z_qeh_layer=None, bb_dZ=None):
        """Set mQEH matrix state and build the q / omega interpolators.

        Single chokepoint for state installation so all three paths
        (synthetic data, restart, fresh QEH) go through the same code.

        Parameters
        ----------
        z_qeh_layer : float or None
            Center-of-the-target-layer z in the qeh frame, used by
            ``calculate_QEH`` to align the inverse-FFT'd DFT pair
            density with the qeh density basis. On the QEH-driven
            path this is ``HS.hs.layers_l[layer].z0`` (exact, by
            construction). On user-supplied-data paths it is not
            generally known; if None, we fall back to the L2-centroid
            of the monopole basis at the smallest q, which equals z0
            for a symmetric basis but can drift for an asymmetric one
            (e.g. when the BB's z-extent leaks past the next layer
            in a bilayer hs).
        bb_dZ : float or None
            The target layer building block's own z-grid spacing
            (``HS.hs.layers_l[layer].bb.dZ``), only used by the
            install-time diagnostic to flag a mismatch between the
            BB's L2-normalization grid and the het z-grid that
            ``S_ab = <rho|rho>`` is computed on. None on user-supplied
            data paths.
        """
        dW = np.asarray(dW_qw_matrix)
        self.dW_qw_matrix = dW
        self.nbasis = int(dW.shape[2])
        self.drho_qzi_target = np.asarray(drho_qzi)
        self.z_z_qeh = np.asarray(z_z_qeh)
        self.dz_qeh = float(dz_qeh)
        self.qqeh_matrix = np.asarray(qqeh_matrix).copy()
        self.wqeh_matrix = np.asarray(wqeh_matrix).copy()
        self.bb_dZ_target = (float(bb_dZ) if bb_dZ is not None else None)
        # Defend against silent unit drift in the restart / synthetic
        # paths.
        if len(self.z_z_qeh) > 1:
            assert np.isclose(self.dz_qeh,
                              self.z_z_qeh[1] - self.z_z_qeh[0]), \
                'dz_qeh inconsistent with z_z_qeh spacing'
        if z_qeh_layer is not None:
            self.z_qeh_layer = float(z_qeh_layer)
        else:
            # Fallback: L2-centroid of monopole basis at smallest q.
            # Accurate for a symmetric basis function; can drift for
            # an asymmetric one. The QEH-driven path passes z_qeh_layer
            # explicitly via calculate_W_QEH, so this fallback only
            # runs for user-supplied-data callers (synthetic tests).
            iq_small = int(np.argmin(np.abs(self.qqeh_matrix)))
            drho_0 = self.drho_qzi_target[iq_small, :, 0]
            w_z = np.abs(drho_0) ** 2
            if w_z.sum() > 0:
                self.z_qeh_layer = float(
                    (w_z * self.z_z_qeh).sum() / w_z.sum())
            else:
                self.z_qeh_layer = float(self.z_z_qeh.mean())
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
        self._log_dW_magnitudes()

    def _log_dW_magnitudes(self):
        """Print the magnitude of dW(q_min, omega=0) so the mQEH matrix
        scale can be compared against the legacy ``GWQEHCorrection``
        scalar dW (which is known to give correct numbers when fed the
        old-style ``-bb`` building blocks via ``qeh.QEH``).

        For each of the three smallest q-values on the qeh grid we print
        the (0, 0) matrix element (the rho-rho-basis analog of the
        legacy monopole-monopole slice), the Frobenius norm of the
        whole nbasis x nbasis block, and -- as a synthetic monopole
        contraction -- ``e_0^T dW e_0`` and ``trace(dW)``, all reported
        in eV. Magnitudes that differ from the legacy scalar dW by
        many orders of magnitude (at the same q, omega) localize the
        10^6 over-estimate to the matrix itself rather than to the
        projection coefficients.
        """
        if self.world.rank != 0:
            return
        dW_qwab = self.dW_qw_matrix          # shape (nq, nw_qeh, nb, nb)
        qqeh = self.qqeh_matrix
        wqeh = self.wqeh_matrix
        iw_static = int(np.argmin(np.abs(wqeh)))   # omega = 0 (or closest)
        sortq = np.argsort(qqeh)
        n_show = min(3, len(qqeh))
        print('mQEH dW-magnitude diagnostic (eV; compare with legacy '
              'GWQEHCorrection scalar dW at same q):', file=self.fd)
        for k in range(n_show):
            iq = int(sortq[k])
            dW_ab = dW_qwab[iq, iw_static]
            mag00 = abs(dW_ab[0, 0]) * Hartree
            mag_fro = float(np.linalg.norm(dW_ab)) * Hartree
            mag_tr = abs(np.trace(dW_ab)) * Hartree
            print(('  iq=%2d  |q|=%.4e Bohr^-1  '
                   '|dW[0,0]|=%.3e  ||dW||_F=%.3e  |tr dW|=%.3e')
                  % (iq, qqeh[iq], mag00, mag_fro, mag_tr),
                  file=self.fd)
        # Also flag a sanity ratio: the (0,0)-vs-Frobenius spread tells
        # whether mQEH spreads dW across many basis modes or
        # concentrates it on the first one (the legacy scalar is
        # effectively just the first mode).
        iq0 = int(sortq[0])
        dW0_ab = dW_qwab[iq0, iw_static]
        denom = float(np.linalg.norm(dW0_ab))
        if denom > 0:
            ratio = abs(dW0_ab[0, 0]) / denom
            print(('  spread: |dW[0,0]| / ||dW||_F = %.3f at iq=%d '
                   '(close to 1 means matrix is monopole-dominated; '
                   'close to 0 means the legacy [0,0] slice is not a '
                   'good proxy for the full mQEH bilinear)')
                  % (ratio, iq0), file=self.fd)
        # Grid-mismatch diagnostic. drho is L2-normalized on the BB's
        # own z-grid (`bb.dZ`). After splining onto the het z-grid
        # (`dz_qeh`), `S_ab = (drho^* drho) * dz_qeh` should still be
        # ~identity along the diagonal *iff* dz_qeh == bb.dZ. If the
        # het builder up/down-samples (dz_qeh != bb.dZ), diag(S)
        # silently picks up a factor of ~ dz_qeh / bb.dZ, which then
        # squares into the d^* W d bilinear via S^{-1}. A factor of
        # (dz_qeh / bb.dZ)^2 of 10^2-10^4 is a plausible component of
        # the 10^6 overshoot.
        if self.bb_dZ_target is not None:
            ratio_dz = self.dz_qeh / self.bb_dZ_target
            print(('  z-grid: dz_qeh=%.6f Bohr, bb.dZ=%.6f Bohr, '
                   'dz_qeh / bb.dZ = %.4f   '
                   '(deviates from 1.0 ==> drho was re-gridded)')
                  % (self.dz_qeh, self.bb_dZ_target, ratio_dz),
                  file=self.fd)
        # diag(S_ab) at the smallest qeh-q: this is the actual numeric
        # value of <rho_a | rho_a> on the het z-grid, computed exactly
        # the same way as inside _calculate_sigma_mqeh. Expect ~1 if
        # the BB normalization survives the het re-grid; otherwise the
        # deviation is the candidate scale of the bilinear's bias.
        drho_za = self.drho_qzi_target[iq0]                # (nz, nb)
        S_ab = (drho_za.conj().T @ drho_za) * self.dz_qeh
        diag_S = np.abs(np.diag(S_ab))
        print(('  diag(S_ab) at iq=%d: [' + ', '.join(
            '%.3e' % v for v in diag_S) + ']   '
              '(expect ~1.0 if drho is L2-normalized on the het grid)')
              % iq0, file=self.fd)
        if diag_S.size > 0:
            print(('  diag(S) deviation: max|1 - diag(S)| = %.3e  '
                   'mean diag(S) = %.3e')
                  % (float(np.max(np.abs(1.0 - diag_S))),
                     float(np.mean(diag_S))),
                  file=self.fd)

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
            wqeh_matrix=wqeh,
            z_qeh_layer=float(target_layer.z0),
            bb_dZ=float(target_layer.bb.dZ))

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

    def _gather_mqeh_query_grid(self):
        """Collect the exact (|q+G_par|, omega) values that the mQEH
        path will query.

        For each GW IBZ q-point we replicate the SingleQPWDescriptor
        construction from ``calculate_QEH`` (with ``self.ecut_mqeh``),
        group the G-vectors by their in-plane component, and harvest
        ``|q + G_par|``. The union across all iq is the q-grid the
        custom mbb building blocks should live on. Returns the
        sorted, strictly-positive q-grid and a reference to
        ``self.omega_w`` (already on the GW frequency grid).
        """
        rcell_cv = 2 * pi * np.linalg.inv(self.gs.gd.cell_cv).T
        q_abs_set = set()
        for q_c in self.qd.ibzk_kc:
            q_v = np.dot(q_c, rcell_cv)
            pd0 = SingleQPWDescriptor.from_q(
                q_c, self.ecut_mqeh, self.gs.gd, gammacentered=True)
            G0_Gv = pd0.get_reciprocal_vectors(add_q=False)
            Gpar_Gv = G0_Gv[:, :2]
            Gpar_rounded = np.round(Gpar_Gv, decimals=8)
            unique_Gpar = np.unique(Gpar_rounded, axis=0)
            for Gpar_v in unique_Gpar:
                qpG_v = q_v[:2] + Gpar_v
                q_abs_set.add(float(np.sqrt(np.sum(qpG_v ** 2))))
        q_q = np.array(sorted(q_abs_set))
        # Drop near-zero entries: the qeh Poisson-1D solver silently
        # replaces q=0 with 1e-12 internally; we handle |q+G_par|=0
        # via the small-q ring in _interpolate_mqeh_data instead.
        q_q = q_q[q_q > 1e-8]
        w_w = np.asarray(self.omega_w)
        return q_q, w_w

    def _build_and_install_mqeh_from_chi(self, *, d, layer):
        """Build per-material mbb files from chi files on the exact
        W query grid, then install the heterostructure dW matrix.

        Avoids spline interpolation in q and omega entirely: every
        ``|q+G_par|`` the GW path will query is a sample point of the
        built BB. Rank 0 writes the mbb files to ``self._bb_output_dir``
        (or ``<filename>_mbb_built/`` when unset); all ranks
        synchronize on a barrier and then load via
        ``qeh.MQEH.heterostructure``.

        See ``GWmQEHCorrection.__init__`` (``chi_files``,
        ``build_bb_on_query_grid``, ``bb_aN``, ``bb_zN``,
        ``bb_output_dir``) for the user-facing knobs.
        """
        import os
        from qeh import MQEH
        from qeh.bb_calculator.bb_builder import interpolate_chi_to_bb
        from qeh.bb_calculator.basis_functions import MQEHBasis

        chi_files = list(self._chi_files)

        d_arr = np.asarray(d, dtype=float)
        if len(d_arr) == len(chi_files) - 1:
            d_arr = interlayer_to_thickness(d_arr)
        assert len(d_arr) == len(chi_files), (
            'GWmQEHCorrection.chi_files must be one entry per layer; '
            'got %d chi files vs %d layer widths'
            % (len(chi_files), len(d_arr)))

        q_q, w_w = self._gather_mqeh_query_grid()
        print(('mQEH exact-grid mode: gathered Nq=%d unique |q+G_par| '
               'values, Nw=%d omega values; building mbb files from '
               '%d distinct chi file(s)')
              % (len(q_q), len(w_w), len(set(chi_files))),
              file=self.fd)

        bb_output_dir = self._bb_output_dir
        if bb_output_dir is None:
            bb_output_dir = self.filename + '_mbb_built'

        chi_to_mbb = {}
        for chi_path in dict.fromkeys(chi_files):
            stem = os.path.splitext(os.path.basename(chi_path))[0]
            if stem.endswith('-chi'):
                stem = stem[:-4]
            chi_to_mbb[chi_path] = os.path.join(
                bb_output_dir, stem + '-mbb')

        if self.world.rank == 0:
            os.makedirs(bb_output_dir, exist_ok=True)
            for chi_path, mbb_path in chi_to_mbb.items():
                print('  %s -> %s' % (chi_path, mbb_path), file=self.fd)
                basis = MQEHBasis(chi_path)
                interpolate_chi_to_bb(
                    chi_path, outfile=mbb_path,
                    aN=self._bb_aN, zN=self._bb_zN,
                    q_grid=q_q, w_grid=w_w,
                    basis=basis)
        self.world.barrier()

        bbfiles = [chi_to_mbb[chi_path] for chi_path in chi_files]
        wmax = float(w_w[-1])

        HS0 = MQEH.heterostructure(
            BBfiles=[bbfiles[layer]],
            layerwidth_l=[d_arr[layer] / Bohr],
            wmax=wmax)
        W0_qwij = HS0.get_screened_potential(subtract_bare_coulomb=True)

        HS = MQEH.heterostructure(
            BBfiles=bbfiles,
            layerwidth_l=d_arr / Bohr,
            wmax=wmax)
        W_qwij = HS.get_screened_potential(subtract_bare_coulomb=True)

        nbasis_target = HS.hs.layers_l[layer].bb.aN
        i0 = sum(HS.hs.layers_l[l].bb.aN for l in range(layer))
        i1 = i0 + nbasis_target
        dW_qwab = (W_qwij[:, :, i0:i1, i0:i1]
                   - W0_qwij[:, :, :nbasis_target, :nbasis_target])

        target_layer = HS.hs.layers_l[layer]
        drho_qzi = np.array(
            [target_layer.get_drho_qza(iq_q=[iq])[0]
             for iq in range(HS.hs.qN)])

        qqeh = HS.hs.q_q.copy()
        wqeh = HS.hs.omega_w.copy()
        self.qqeh = qqeh
        self.wqeh = wqeh

        self._install_mqeh_matrix(
            dW_qw_matrix=dW_qwab,
            drho_qzi=drho_qzi,
            z_z_qeh=HS.hs.z_z.copy(),
            dz_qeh=HS.hs.dz,
            qqeh_matrix=qqeh,
            wqeh_matrix=wqeh,
            z_qeh_layer=float(target_layer.z0),
            bb_dZ=float(target_layer.bb.dZ))

        if self.world.rank == 0:
            np.savez(self.filename + '_dW_qw.npz',
                     qqeh=qqeh, wqeh=wqeh,
                     dW_qw_matrix=dW_qwab, drho_qzi=drho_qzi,
                     z_z_qeh=self.z_z_qeh, dz_qeh=self.dz_qeh,
                     nbasis=self.nbasis)

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

        Returns array of shape (nw_gw, nbasis, nbasis).

        Out-of-range handling:
          * |q| <= self._q0_cut: substitute the small-q ring average
            (or zero when include_q0=False), as in the parent's get_W_on_grid.
          * |q| < qqeh.min(): if the q0 averaging didn't fire (no
            small-q ring available), fall back to the spline value at
            qqeh.min() rather than a cubic extrapolation downward.
          * |q| > qqeh.max(): physical dW for an interlayer screening
            correction decays roughly as exp(-q d) at large q, so the
            cubic-spline polynomial extrapolation is unbounded and
            wrong by many orders of magnitude (and typically with a
            wrong sign as the cubic flips). Clamp to zero in this
            regime instead -- a tiny missing tail is far less harmful
            than a polynomial that blows up. The one-shot warning
            still fires so the user can grow qqeh.max() or shrink
            ecut_mqeh to push the cutoff out.
        """
        if (self._q0_dW_Wab is not None
                and q_abs <= self._q0_cut):
            return self._q0_dW_Wab.copy()
        if q_abs > self._qqeh_max:
            if not getattr(self, '_warned_qmax', False):
                print(('WARNING: evaluating Delta-W at |q+G_par| > '
                       'qqeh.max()=%.3f Bohr^-1; clamping dW to 0 for '
                       'q > qqeh.max() (physical W decays exponentially '
                       'at large q, but cubic-spline extrapolation '
                       'diverges polynomially and would dominate the '
                       'sum). Consider increasing the mQEH q_max in '
                       'the building block or decreasing ecut_mqeh.')
                      % self._qqeh_max, file=self.fd)
                self._warned_qmax = True
            nb = self.nbasis
            nw_gw = len(self.omega_w)
            return np.zeros((nw_gw, nb, nb), dtype=complex)
        # |q| inside [qqeh.min(), qqeh.max()] is the interpolation regime.
        # For |q| < qqeh.min() (and outside the q0_cut block above), the
        # spline would extrapolate downward; pin to the qqeh.min() value
        # instead so a divergent dW(q -> 0) doesn't get amplified by a
        # cubic that goes the wrong way.
        q_eval = max(float(q_abs), float(self._qqeh_sorted[0]))
        return (self._dW_spline_re(q_eval)
                + 1j * self._dW_spline_im(q_eval))

    def _eval_drho(self, q_abs):
        """Evaluate density basis functions at |q|.

        Returns array of shape (nz, nbasis). Clamps |q| to the
        ``qqeh`` interpolation range to avoid the same divergent
        cubic-extrapolation pathology fixed in
        :meth:`_eval_dW_on_gwgrid`. The induced-density basis
        functions also live on the qqeh grid only, and extrapolating
        them past the endpoints can produce nonsense the rho-LS
        projection then propagates into the self-energy.
        """
        q_clip = float(
            np.clip(q_abs, self._qqeh_sorted[0], self._qqeh_sorted[-1]))
        return (self._drho_spline_re(q_clip)
                + 1j * self._drho_spline_im(q_clip))

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

        # Reset LS-residual diagnostic state. The check fires inside
        # _calculate_sigma_mqeh and emits per-occurrence warnings up to
        # _residual_max_warnings; after that it accumulates silently and
        # we print a summary below.
        self._residual_warnings_emitted = 0
        self._residual_max_warnings = 10
        self._residual_threshold = 0.25
        self._residual_max_seen = 0.0
        self._residual_count_over_threshold = 0
        self._residual_count_total = 0

        # Per-sample dW / d_a / bilinear diagnostic prints. Fires once
        # (first call to _calculate_sigma_mqeh with non-empty data),
        # then suppresses. Compare the printed magnitudes against the
        # legacy GWQEHCorrection scalar dW and against |n_G|^2 at the
        # same q-point to localize the 10^6 overshoot.
        self._diag_sample_printed = False

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

        # ----- z-frame alignment between the DFT cell and the qeh grid -----
        # The pair density n_G is computed in the DFT cell with the cell
        # origin at z=0 and atoms at their cell-relative positions, so the
        # inverse z-FFT
        #     bar_rho(z) = (1/Lz) sum_{Gz} n_G(Gpar, Gz) e^{iGz z}
        # gives bar_rho as a function of z *in the DFT-cell frame*. It
        # peaks at z_DFT_layer, the centroid of the target layer's atoms
        # in the cell (~ Lz/2 for a centered slab).
        #
        # The qeh density basis drho_a(z) lives on the QEH heterostructure
        # z-grid, where the target layer sits at z_qeh_layer
        # (= HS.hs.layers_l[layer].z0, typically a few Bohr). The qeh
        # z-grid is built by QEH from layer thicknesses + BB extents and
        # has no notion of the DFT cell origin.
        #
        # Without correction the projection R_a = <drho_a | bar_rho>
        # samples drho_a where it peaks (z_qeh = z_qeh_layer) but
        # bar_rho_DFT at those same numerical z values is deep in the
        # vacuum of the DFT cell, so the integral is exponentially
        # suppressed (~exp(-(z_DFT_layer-z_qeh_layer)^2 / sigma^2),
        # often 10^-10 or smaller). The fix is to evaluate the inverse
        # FFT at z + z_offset, with z_offset = z_DFT_layer - z_qeh_layer
        # -- equivalently, multiply n_G by e^{iGz z_offset} before
        # summing; via the shift theorem these are the same operation.
        pos_av = self.gs.get_pos_av()              # Bohr
        z_DFT_layer = float(pos_av[:, 2].mean())
        z_offset = z_DFT_layer - self.z_qeh_layer
        n_periodic_images = (self.z_z_qeh[-1] - self.z_z_qeh[0]) / L
        print(f'mQEH z-frame: z_DFT_layer={z_DFT_layer:.4f} Bohr, '
              f'z_qeh_layer={self.z_qeh_layer:.4f} Bohr, '
              f'z_offset={z_offset:.4f} Bohr',
              file=self.fd)
        print(f'mQEH grids: Lz_DFT={L:.4f} Bohr, '
              f'qeh z-grid extent [{self.z_z_qeh[0]:.3f}, '
              f'{self.z_z_qeh[-1]:.3f}] Bohr ({n_periodic_images:.2f} '
              f'DFT periods)',
              file=self.fd)
        if n_periodic_images > 1.5:
            print(('  NOTE: the qeh z-grid covers more than one DFT '
                   'cell. The inverse z-FFT of the DFT pair density '
                   'is periodic with period Lz_DFT, so multiple '
                   'periodic images of the layer peak will appear in '
                   'the qeh grid. The projection only captures one of '
                   'them, inflating the LS residual; the contracted '
                   'self-energy is unaffected as long as drho doesn''t '
                   'overlap with the periodic images.'),
                  file=self.fd)

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
                # NB: no `*= L` here -- the spatial prefactor in
                # _calculate_sigma_mqeh is now 1/(N_q*2pi*A) directly,
                # not the parent class's 1/(N_q*2pi*V)*L combo. This is
                # algebraically identical (V = A*L) but exposes the
                # 2D-area assumption so the legacy-vs-mQEH unit
                # convention is auditable in one place.
                dW_wab = self._eval_dW_on_gwgrid(q_abs)

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
            #
            # The +z_offset shift moves the inverse-FFT evaluation point
            # from the qeh frame into the DFT-cell frame, so bar_rho
            # peaks at the qeh-frame layer center (z = z_qeh_layer)
            # where drho is, rather than at the DFT-cell layer center
            # (where drho would see only its tail).
            Q_G = pd0.Q_qG[0]
            i_cG = np.array(np.unravel_index(Q_G, N_c))
            Gz_idx = i_cG[2]
            Gz_idx_wrapped = np.where(Gz_idx > N_c[2] // 2,
                                      Gz_idx - N_c[2], Gz_idx)
            z_qeh_shifted = self.z_z_qeh + z_offset
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
                    np.exp(1j * np.outer(z_qeh_shifted, Gz_values)))

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
                            drho_Gpar_za, S_Gpar_ab, L,
                            qplusGpar_abs=qplusGpar_abs)

                        nn = kpt1.n1 + n - self.bands[0]
                        self.sigma_sin[kpt1.s, i, nn] += sigma
                        self.dsigma_sin[kpt1.s, i, nn] += dsigma

        self.world.sum(self.sigma_sin)
        self.world.sum(self.dsigma_sin)

        # LS-residual summary. The residual measures how well the mQEH
        # basis can represent the in-plane pair density that the self-
        # energy sandwiches; a small value means the bilinear
        # d^dagger W d is a faithful expansion of <rho|W|rho>, a large
        # value means the formula is contracting a heavily-truncated
        # density and the answer is unreliable.
        if self._residual_count_total > 0:
            frac = (self._residual_count_over_threshold
                    / self._residual_count_total)
            print(('mQEH LS-residual summary: '
                   '%d / %d (m, G_par) pair densities exceeded the '
                   '%.2f threshold (%.1f%%); max ||rho - rho_mqeh|| / '
                   '||rho|| = %.4f')
                  % (self._residual_count_over_threshold,
                     self._residual_count_total,
                     self._residual_threshold,
                     100.0 * frac,
                     self._residual_max_seen),
                  file=self.fd)

        # Pair-density dump (opt-in via dump_pair_densities=True).
        if self._dump_pair_densities and self._pair_density_samples:
            self._save_pair_density_dump()

        self.complete = True
        self.save_state_file()

        return self.sigma_sin, self.dsigma_sin

    def _save_pair_density_dump(self):
        """Write the stashed pair-density / basis / LS-coefficient
        samples to ``<filename>_mqeh_pair_densities.npz``.

        The companion script
        ``gpaw/response/gwmqeh_plot_pair_densities.py`` loads this file
        and plots each pair density alongside its rho-LS fit, so the
        alignment between the (qeh-frame-shifted) DFT pair density and
        the mQEH basis can be visualized cheaply without GPAW on a
        laptop.
        """
        if self.world.rank != 0:
            return
        samples = self._pair_density_samples
        n = len(samples)
        rho_mz = np.array([s['rho_mz'] for s in samples])     # (n, nz)
        drho_za = np.array([s['drho_za'] for s in samples])   # (n, nz, nb)
        d_a = np.array([s['d_a'] for s in samples])           # (n, nb)
        iq = np.array([s['iq'] for s in samples], dtype=int)
        ig = np.array([s['ig'] for s in samples], dtype=int)
        m = np.array([s['m'] for s in samples], dtype=int)
        qabs = np.array([s['qplusGpar_abs'] for s in samples],
                        dtype=float)
        fname = self.filename + '_mqeh_pair_densities.npz'
        np.savez(
            fname,
            z_z_qeh=self.z_z_qeh,
            dz_qeh=self.dz_qeh,
            z_qeh_layer=self.z_qeh_layer,
            rho_mz=rho_mz,
            drho_za=drho_za,
            d_a=d_a,
            iq=iq,
            ig=ig,
            m=m,
            qplusGpar_abs=qabs,
        )
        print(f'Wrote {n} pair-density samples to {fname}',
              file=self.fd)

    def _check_LS_residual(self, rho_mz, d_a, drho_za, ig,
                           qplusGpar_abs=None):
        """Diagnostic: compare the rho-LS reconstruction against the
        actual in-plane pair density.

        We compute
            rho_mqeh(z) = sum_alpha d_alpha rho_alpha(z)
        for each band ``m`` and compare to the original
        ``rho_mz(z) = (1/Lz) sum_Gz n(G_par, Gz) e^{iGz z}`` using the
        relative L2 norm

            residual = ||rho_mz - rho_mqeh||_2 / ||rho_mz||_2.

        A faithful expansion has residual << 1. A residual close to 1
        means the basis does not span the pair density at all -- typical
        causes are (i) a frame mismatch between the qeh z-grid and the
        DFT cell origin (rho_mz peaks where drho has no support), or
        (ii) a basis with too few functions to capture the pair
        density's z-shape. Per-occurrence warnings are emitted up to a
        cap so we don't drown stdout; a final summary prints in
        ``calculate_QEH``.
        """
        rho_recon = d_a @ drho_za.T
        diff = rho_mz - rho_recon
        norms_rho = np.linalg.norm(rho_mz, axis=1)
        norms_diff = np.linalg.norm(diff, axis=1)
        mask = norms_rho > 1e-20
        if not mask.any():
            return
        residuals = np.zeros_like(norms_rho)
        residuals[mask] = norms_diff[mask] / norms_rho[mask]

        self._residual_count_total += int(mask.sum())
        bad = mask & (residuals > self._residual_threshold)
        n_bad = int(bad.sum())
        if n_bad == 0:
            self._residual_max_seen = max(self._residual_max_seen,
                                          float(residuals[mask].max()))
            return

        self._residual_count_over_threshold += n_bad
        worst_m = int(np.argmax(residuals))
        worst = float(residuals[worst_m])
        self._residual_max_seen = max(self._residual_max_seen, worst)

        if self._residual_warnings_emitted < self._residual_max_warnings:
            qabs_str = (
                f', |q+G_par|={qplusGpar_abs[ig]:.4f} Bohr^-1'
                if qplusGpar_abs is not None else '')
            iq = getattr(self, 'nq', -1)
            print(('WARNING: large LS residual %.3f at iq=%d, ig=%d, '
                   'm=%d%s -- the mQEH basis cannot reproduce the '
                   'in-plane pair density at this (q+G_par). If many '
                   'of these fire, the d^dagger W d contraction is '
                   'integrating over a heavily-truncated rho and the '
                   'self-energy is unreliable. A common cause is a '
                   'frame mismatch between the qeh z-grid (drho peaks '
                   'at z_qeh_layer) and the DFT cell origin (pair '
                   'density peaks at z_DFT_layer).')
                  % (worst, iq, ig, worst_m, qabs_str),
                  file=self.fd)
            self._residual_warnings_emitted += 1
            if (self._residual_warnings_emitted
                    == self._residual_max_warnings):
                print('  (further LS-residual warnings suppressed; '
                      'see summary at end of calculate_QEH)',
                      file=self.fd)

    def _calculate_sigma_mqeh(self, n_mG, deps_m, f_m, Wpm_Gpar,
                              G_indices_per_Gpar, phase_zg_per_Gpar,
                              drho_Gpar_za, S_Gpar_ab, Lz,
                              qplusGpar_abs=None):
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
            For each unique G_parallel group, exp(i Gz (z_qeh + z_offset))
            on the QEH z-grid with shape (nz_qeh, n_gz). z_offset is the
            DFT-to-qeh z alignment computed in calculate_QEH; the phase
            already encodes it so that the inverse FFT of n_G yields a
            pair density aligned with drho. None for empty groups.
        drho_Gpar_za : ndarray (n_Gpar, nz_qeh, nbasis)
            Density basis functions rho_alpha(|q+G_par|; z).
        S_Gpar_ab : ndarray (n_Gpar, nbasis, nbasis)
            Rho overlap matrix S_{ab} = int rho_a* rho_b dz on the het
            z-grid.
        Lz : float
            DFT cell height (for inverse-FFT normalization).
        qplusGpar_abs : ndarray (n_Gpar,) or None
            |q + G_par| in Bohr^-1 for each G_par group, only used in
            LS-residual warnings to make them actionable.
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
        # Spatial prefactor 1/(N_q*2pi*A) (Eq.(9) of W&T 2017). The
        # parent's scalar path uses 1/(N_q*2pi*V) and multiplies dW
        # by L; that produces the same x*L = 1/(N_q*2pi*A). The mQEH
        # path now folds the L into x directly so the 1/A assumption
        # is local to one line.
        A = abs(np.linalg.det(self.gs.gd.cell_cv[:2, :2]))
        x = 1.0 / (self.qd.nbzkpts * 2 * pi * A)

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
            if self._lagrange_constrained_ls:
                # Constrained LS: minimize ||bar_rho - sum_a d_a rho_a||
                # subject to (int rho_a dz) d_a = int bar_rho dz. The
                # constraint is the charge-conservation identity that
                # forces the truncation residual to have zero z-integral,
                # preserving the cancellation of the 1/q Coulomb
                # divergence in d^* W d as q -> 0. KKT system follows
                # qeh/bb_calculator/bb_builder.py:_constrained_ls and
                # the (commented-out) Lagrange-multiplier branch of
                # qehbse.WqzQEH.get_projector_overlap.
                # c_a = int rho_a(z) dz (no conjugate); the physical
                # constraint is sum_a d_a c_a = int bar_rho dz, so the
                # KKT bottom row is c^T d = t and the right column
                # carries the gradient of the constraint w.r.t. d^*,
                # which is c^* (cf. qeh/bb_calculator/bb_builder.py
                # _constrained_ls).
                c_a = drho_za.sum(0) * self.dz_qeh          # (nb,)
                t_m = rho_mz.sum(1) * self.dz_qeh           # (nbands,)
                KKT = np.zeros((nb + 1, nb + 1), dtype=complex)
                KKT[:nb, :nb] = S_Gpar_ab[ig]
                KKT[:nb, nb] = c_a.conj()
                KKT[nb, :nb] = c_a
                # KKT[nb, nb] = 0  (already zero)
                rhs = np.zeros((rho_mz.shape[0], nb + 1), dtype=complex)
                rhs[:, :nb] = R_m_a
                rhs[:, nb] = t_m
                sol = np.linalg.solve(KKT, rhs.T).T          # (nbands, nb+1)
                d_a = sol[:, :nb]
            else:
                # Unconstrained normal equation: S @ d.T = R.T
                # Kept available for A/B testing against the constrained
                # solve above; see __init__'s lagrange_constrained_ls.
                d_a = np.linalg.solve(S_Gpar_ab[ig], R_m_a.T).T
            d_mGpar_a[:, ig, :] = d_a
            # Diagnostic: how well does the rho-LS expansion reproduce
            # the pair density? See _check_LS_residual.
            self._check_LS_residual(rho_mz, d_a, drho_za, ig,
                                    qplusGpar_abs=qplusGpar_abs)
            # Optional pair-density dump: stash up to _dump_max
            # (rho_mz, drho_za, d_a) tuples for offline plotting via
            # gwmqeh_plot_pair_densities.py.
            if (self._dump_pair_densities
                    and len(self._pair_density_samples) < self._dump_max):
                qabs = (float(qplusGpar_abs[ig])
                        if qplusGpar_abs is not None else float('nan'))
                for m in range(rho_mz.shape[0]):
                    if len(self._pair_density_samples) >= self._dump_max:
                        break
                    self._pair_density_samples.append({
                        'rho_mz': np.asarray(rho_mz[m]).copy(),
                        'drho_za': np.asarray(drho_za).copy(),
                        'd_a': np.asarray(d_a[m]).copy(),
                        'iq': int(getattr(self, 'nq', -1)),
                        'ig': int(ig),
                        'm': int(m),
                        'qplusGpar_abs': qabs,
                    })

        # Prefactor x = 1/(N_q*2pi*A) -- set above; dW has no extra L
        # multiplier in this path. Matches Eq.(9) of W&T 2017.
        sigma = 0.0
        dsigma = 0.0

        # One-shot per-sample diagnostic. We print the rho-LS
        # coefficient magnitude, the dW magnitude and the bilinear
        # d^dagger W d for the first (m, ig) we encounter on this
        # rank. These three numbers, together with the prefactor x
        # printed below, fully determine where a 10^6 overshoot
        # could be hiding (matrix scale vs. coefficient scale vs.
        # prefactor).
        diag_fire = (not self._diag_sample_printed
                     and self.world.rank == 0
                     and d_mGpar_a.shape[0] > 0
                     and n_Gpar > 0)
        if diag_fire:
            m_diag = 0
            ig_diag = (int(np.argmin(qplusGpar_abs))
                       if qplusGpar_abs is not None else 0)
            d_diag = d_mGpar_a[m_diag, ig_diag]
            iw_static = 0       # first GW omega slot (~ omega = 0)
            W_diag = Wpm_Gpar[ig_diag, iw_static]
            bilinear = complex(d_diag.conj() @ W_diag @ d_diag)
            qabs_str = (
                '%.4e Bohr^-1' % float(qplusGpar_abs[ig_diag])
                if qplusGpar_abs is not None else 'unknown')
            print(('mQEH sample diagnostic (rank 0, first non-empty '
                   'call): ig=%d, |q+G_par|=%s, m=%d')
                  % (ig_diag, qabs_str, m_diag), file=self.fd)
            print(('  |d_a|     = [' + ', '.join(
                '%.3e' % abs(z) for z in d_diag) + ']'), file=self.fd)
            print(('  |d_a|_2   = %.3e   (rho-LS norm of the pair density)')
                  % float(np.linalg.norm(d_diag)), file=self.fd)
            S_diag = np.abs(np.diag(S_Gpar_ab[ig_diag]))
            print(('  diag(S)   = [' + ', '.join(
                '%.3e' % v for v in S_diag) + ']   '
                  '(rho overlap on het grid; ~1.0 if BB norm survives)'),
                  file=self.fd)
            # Charge-conservation diagnostic: sum_a d_a * (int rho_a dz)
            # should equal int bar_rho dz. Print both sides and the
            # residual. With lagrange_constrained_ls=True the residual
            # should be ~ numerical zero; with False it can be O(1)
            # and is the suspected source of the small-q overshoot.
            drho_za_diag = drho_Gpar_za[ig_diag]
            c_a_diag = drho_za_diag.sum(0) * self.dz_qeh
            cTd = complex(c_a_diag @ d_diag)
            # rho_mz for band m=0, ig=ig_diag isn't directly available
            # here (we only have d_mGpar_a). Recompute t from the
            # original probe via the inverse FFT for the sample only.
            G_indices_diag = G_indices_per_Gpar[ig_diag]
            if (len(G_indices_diag) > 0
                    and phase_zg_per_Gpar[ig_diag] is not None):
                rho_mz_diag = (
                    n_mG[m_diag:m_diag + 1, G_indices_diag]
                    @ phase_zg_per_Gpar[ig_diag].T / Lz)
                t_diag = complex(rho_mz_diag.sum() * self.dz_qeh)
            else:
                t_diag = 0.0 + 0.0j
            print(('  charge-cons: c^T d = %.3e   '
                   'int bar_rho dz = %.3e   '
                   '|c^T d - t| = %.3e   '
                   '(constraint = %s)')
                  % (abs(cTd), abs(t_diag), abs(cTd - t_diag),
                     'ON' if self._lagrange_constrained_ls else 'OFF'),
                  file=self.fd)
            print(('  ||W||_F   = %.3e Ha*Bohr^2   '
                   '|W[0,0]| = %.3e Ha*Bohr^2')
                  % (float(np.linalg.norm(W_diag)),
                     abs(W_diag[0, 0])), file=self.fd)
            print(('  |d^* W d| = %.3e Ha*Bohr^2 (compare to legacy '
                   "GWQEHCorrection's |n_G[0]|^2 * dW_legacy at same q)")
                  % abs(bilinear), file=self.fd)
            print(('  prefactor x = %.3e Bohr^-2 (1/(N_q*2pi*A); '
                   'final per-sample contribution magnitude '
                   '~ x * |d^* W d| = %.3e Ha)')
                  % (x, x * abs(bilinear)), file=self.fd)
            self._diag_sample_printed = True

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
