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
        self.context = ResponseContext(txt=filename + '.txt', comm=world)
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

        # Calculate screened potential of Heterostructure
        if dW_qw is None:
            if restart:
                try:
                    data = np.load(filename + "_dW_qw.npz")
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
            ('Frequency grids doesnt match!')

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
        nibzk = self.gs.kd.nibzkpts
        for i, k in enumerate(self.kpts):
            for s in range(self.nspins):
                u = s * nibzk + k
                kpt = self.gs.kpt_u[u]
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

            # Screened potential
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
            if (data['kpts'] == self.kpts and
                (data['bands'] == self.bands).all() and
                    data['nbands'] == self.nbands):
                self.nq = data['last_q']
                self.complete = data['complete']
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
        from qeh import QEH
        from qeh.heterostructure import expand_layers

        structure = expand_layers(structure)
        self.w_grid = self.omega_w
        wmax = self.w_grid[-1]
        # qmax = (self.q_grid).max()

        # Single layer
        if len(d) == len(structure) - 1:
            d = interlayer_to_thickness(d)
        HS0 = QEH.heterostructure(
            BBfiles=[structure[layer]],
            layerwidth_l=[d[layer] / Bohr],
            wmax=wmax,
            # qmax=qmax / Bohr
        )

        W0_qw = HS0.get_screened_potential()[..., 0, 0]

        # Full heterostructure

        HS = QEH.heterostructure(BBfiles=structure, layerwidth_l=d / Bohr,
                                 wmax=wmax,
                                 # qmax=qmax / Bohr
                                 )
        basis_idx = 2 * layer
        W_qw = HS.get_screened_potential()[..., basis_idx, basis_idx]

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
                 dW_qw=None, qqeh=None, wqeh=None,
                 txt=sys.stdout, world=mpi.world, domega0=0.025,
                 omega2=10.0, eta=0.1, include_q0=True, metal=False,
                 restart=False, ecut_mqeh=50.0,
                 dW_qw_matrix=None, phi_qiz=None, drho_qzi=None,
                 z_z_qeh=None, dz_qeh=None):

        self.ecut_mqeh = ecut_mqeh / Hartree

        # These will be set by calculate_W_QEH or provided directly
        self.dW_qw_matrix = None
        self.phi_qiz = None
        self.drho_qzi = None
        self.nbasis = None
        self.layer_index = layer
        self.qqeh_matrix = None
        self.wqeh_matrix = None

        # Store user-provided mQEH data (if any) for setup after parent init
        self._init_dW_qw_matrix = dW_qw_matrix
        self._init_phi_qiz = phi_qiz
        self._init_drho_qzi = drho_qzi
        self._init_z_z_qeh = z_z_qeh
        self._init_dz_qeh = dz_qeh

        # Call parent __init__, which will call calculate_W_QEH
        # (or load dW_qw) and set up everything
        super().__init__(
            calc=calc, gwfile=gwfile, filename=filename, kpts=kpts,
            bands=bands, structure=structure, d=d, layer=layer,
            dW_qw=dW_qw, qqeh=qqeh, wqeh=wqeh,
            txt=txt, world=world, domega0=domega0,
            omega2=omega2, eta=eta, include_q0=include_q0,
            metal=metal, restart=restart)

        # If full mQEH data was provided directly, install it now
        if self._init_dW_qw_matrix is not None:
            self.dW_qw_matrix = self._init_dW_qw_matrix
            self.nbasis = self._init_dW_qw_matrix.shape[2]
            self.qqeh_matrix = qqeh.copy()
            self.wqeh_matrix = wqeh.copy()
            self.phi_qiz_target = self._init_phi_qiz
            self.drho_qzi_target = self._init_drho_qzi
            self.z_z_qeh = self._init_z_z_qeh
            self.dz_qeh = self._init_dz_qeh
            self._interpolate_mqeh_data()

    def calculate_W_QEH(self, structure, d, layer=0):
        """Calculate the full mQEH Delta-W matrix.

        Returns the scalar (monopole) dW_qw for the parent class grid
        interpolation, but also stores the full basis Delta-W matrix,
        density basis functions, and potential basis functions needed
        for the mQEH projection.
        """
        from qeh import QEH
        from qeh.heterostructure import expand_layers

        structure = expand_layers(structure)
        self.w_grid = self.omega_w
        wmax = self.w_grid[-1]

        if len(d) == len(structure) - 1:
            d = interlayer_to_thickness(d)

        # Single layer (isolated monolayer)
        HS0 = QEH.heterostructure(
            BBfiles=[structure[layer]],
            layerwidth_l=[d[layer] / Bohr],
            wmax=wmax,
        )

        W0_qwij = HS0.get_screened_potential()

        # Full heterostructure
        HS = QEH.heterostructure(
            BBfiles=structure,
            layerwidth_l=d / Bohr,
            wmax=wmax,
        )

        W_qwij = HS.get_screened_potential()

        # Number of basis functions for the target layer
        nbasis_target = HS.hs.layers_l[layer].bb.aN

        # Extract the block of W corresponding to the target layer
        i0 = sum(HS.hs.layers_l[l].bb.aN for l in range(layer))
        i1 = i0 + nbasis_target

        # Delta-W for the target layer block
        # Shape: (nq, nw, nbasis, nbasis)
        dW_qwab = (W_qwij[:, :, i0:i1, i0:i1]
                   - W0_qwij[:, :, :nbasis_target, :nbasis_target])

        # Store the full mQEH data
        self.dW_qw_matrix = dW_qwab
        self.nbasis = nbasis_target
        self.qqeh_matrix = HS.hs.q_q.copy()
        self.wqeh_matrix = HS.hs.omega_w.copy()

        # Store density and potential basis functions for the target layer
        # drho_qzi: density basis functions on z-grid, shape (nq, nz, nbasis)
        # phi_qiz: potential basis functions, shape (nq, nbasis, nz)
        target_layer = HS.hs.layers_l[layer]
        self.drho_qzi_target = np.array(
            [target_layer.get_drho_qza(iq_q=[iq])[0]
             for iq in range(HS.hs.qN)])
        self.phi_qiz_target = np.array(
            [target_layer.get_phi_qaz(iq_q=[iq])[0]
             for iq in range(HS.hs.qN)])
        self.z_z_qeh = HS.hs.z_z.copy()
        self.dz_qeh = HS.hs.dz

        self.wqeh = HS.hs.omega_w
        self.qqeh = HS.hs.q_q

        # Save for restart
        if self.world.rank == 0:
            data = {'qqeh': self.qqeh,
                    'wqeh': self.wqeh,
                    'dW_qw': dW_qwab[:, :, 0, 0],  # monopole for compat
                    'dW_qw_matrix': dW_qwab,
                    'drho_qzi': self.drho_qzi_target,
                    'phi_qiz': self.phi_qiz_target,
                    'z_z_qeh': self.z_z_qeh,
                    'dz_qeh': self.dz_qeh,
                    'nbasis': self.nbasis}
            np.savez(self.filename + "_dW_qw.npz", **data)

        # Return monopole component for parent class
        return dW_qwab[:, :, 0, 0]

    def get_W_on_grid(self, dW_qw, include_q0=True, metal=False):
        """Interpolate the full mQEH Delta-W matrix onto the GW grid.

        The parent class method is called for the monopole scalar dW.
        Here we additionally interpolate the full Delta-W matrix and
        the basis functions onto the q-grid used in the GW calculation.
        """
        # Call parent to get the scalar dW on the GW grid
        # (used as fallback / comparison)
        dW_scalar = super().get_W_on_grid(dW_qw, include_q0=include_q0,
                                          metal=metal)

        # If we have the full matrix data, interpolate it too
        if self.dW_qw_matrix is not None:
            self._interpolate_mqeh_data()

        return dW_scalar

    def _interpolate_mqeh_data(self):
        """Pre-compute interpolators for the mQEH Delta-W matrix and
        basis functions so they can be evaluated at arbitrary |q+G_par|."""
        from scipy.interpolate import CubicSpline

        qqeh = self.qqeh_matrix
        sortq = np.argsort(qqeh)
        self._qqeh_sorted = qqeh[sortq]

        # Interpolate Delta-W matrix: shape (nq, nw, nbasis, nbasis)
        dW_sorted = self.dW_qw_matrix[sortq]
        nb = self.nbasis
        nw = dW_sorted.shape[1]

        # Build spline interpolators for each (alpha, beta, w) component
        # Store as array of splines for efficiency
        self._dW_splines_real = np.empty((nb, nb, nw), dtype=object)
        self._dW_splines_imag = np.empty((nb, nb, nw), dtype=object)
        for a in range(nb):
            for b in range(nb):
                for iw in range(nw):
                    vals = dW_sorted[:, iw, a, b]
                    self._dW_splines_real[a, b, iw] = CubicSpline(
                        self._qqeh_sorted, vals.real, extrapolate=True)
                    self._dW_splines_imag[a, b, iw] = CubicSpline(
                        self._qqeh_sorted, vals.imag, extrapolate=True)

        # Interpolate density basis functions: shape (nq, nz, nbasis)
        drho_sorted = self.drho_qzi_target[sortq]
        nz = drho_sorted.shape[1]
        self._drho_splines_real = np.empty((nz, nb), dtype=object)
        self._drho_splines_imag = np.empty((nz, nb), dtype=object)
        for iz in range(nz):
            for a in range(nb):
                vals = drho_sorted[:, iz, a]
                self._drho_splines_real[iz, a] = CubicSpline(
                    self._qqeh_sorted, vals.real, extrapolate=True)
                self._drho_splines_imag[iz, a] = CubicSpline(
                    self._qqeh_sorted, vals.imag, extrapolate=True)

        # Interpolate potential basis functions: shape (nq, nbasis, nz)
        phi_sorted = self.phi_qiz_target[sortq]
        self._phi_splines_real = np.empty((nb, nz), dtype=object)
        self._phi_splines_imag = np.empty((nb, nz), dtype=object)
        for a in range(nb):
            for iz in range(nz):
                vals = phi_sorted[:, a, iz]
                self._phi_splines_real[a, iz] = CubicSpline(
                    self._qqeh_sorted, vals.real, extrapolate=True)
                self._phi_splines_imag[a, iz] = CubicSpline(
                    self._qqeh_sorted, vals.imag, extrapolate=True)

        self._nw_qeh = nw
        self._nz_qeh = nz

    def _eval_dW_matrix(self, q_abs):
        """Evaluate the Delta-W matrix at a given |q| value.

        Returns array of shape (nw_qeh, nbasis, nbasis).
        """
        nb = self.nbasis
        nw = self._nw_qeh
        dW_wab = np.empty((nw, nb, nb), dtype=complex)
        for a in range(nb):
            for b in range(nb):
                for iw in range(nw):
                    dW_wab[iw, a, b] = (
                        self._dW_splines_real[a, b, iw](q_abs)
                        + 1j * self._dW_splines_imag[a, b, iw](q_abs))
        return dW_wab

    def _eval_phi(self, q_abs):
        """Evaluate potential basis functions at |q|.

        Returns array of shape (nbasis, nz).
        """
        nb = self.nbasis
        nz = self._nz_qeh
        phi_az = np.empty((nb, nz), dtype=complex)
        for a in range(nb):
            for iz in range(nz):
                phi_az[a, iz] = (
                    self._phi_splines_real[a, iz](q_abs)
                    + 1j * self._phi_splines_imag[a, iz](q_abs))
        return phi_az

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
        nibzk = self.gs.kd.nibzkpts
        for i, k in enumerate(self.kpts):
            for s in range(self.nspins):
                u = s * nibzk + k
                kpt = self.gs.kpt_u[u]
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
                # Evaluate Delta-W at this |q+G_par| and interpolate
                # to the GW frequency grid
                dW_wab = self._eval_and_interpolate_dW(q_abs)
                # dW_wab has shape (nw, nbasis, nbasis)
                # Apply L factor (unit cell height) as in parent
                dW_wab *= L

                # Set up Wpm for Hilbert transform
                Wpm_Gpar[ig, :nw] = dW_wab
                Wpm_Gpar[ig, nw:] = dW_wab.copy()

                # Apply Hilbert transforms
                self.htp(Wpm_Gpar[ig, :nw])
                self.htm(Wpm_Gpar[ig, nw:])

            # Prepare potential basis functions for projection
            # at each |q + G_parallel|
            # phi_az(|q+G_par|) has shape (nbasis, nz_qeh)
            phi_Gpar_az = np.zeros(
                (n_Gpar, nb, self._nz_qeh), dtype=complex)
            for ig in range(n_Gpar):
                phi_Gpar_az[ig] = self._eval_phi(qplusGpar_abs[ig])

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
                            inverse_idx, N_c, pd0, phi_Gpar_az,
                            qplusGpar_abs, A)

                        nn = kpt1.n1 + n - self.bands[0]
                        self.sigma_sin[kpt1.s, i, nn] += sigma
                        self.dsigma_sin[kpt1.s, i, nn] += dsigma

        self.world.sum(self.sigma_sin)
        self.world.sum(self.dsigma_sin)

        self.complete = True
        self.save_state_file()

        return self.sigma_sin, self.dsigma_sin

    def _eval_and_interpolate_dW(self, q_abs):
        """Evaluate Delta-W matrix at |q| and interpolate to GW freq grid.

        Returns array of shape (nw, nbasis, nbasis) on the GW frequency
        grid self.omega_w.
        """
        from scipy.interpolate import CubicSpline

        # Evaluate on the QEH frequency grid
        dW_wab_qeh = self._eval_dW_matrix(q_abs)
        wqeh = self.wqeh_matrix

        nb = self.nbasis
        nw = self.nw
        w_grid = self.omega_w

        dW_wab = np.zeros((nw, nb, nb), dtype=complex)
        for a in range(nb):
            for b in range(nb):
                vals = dW_wab_qeh[:, a, b]
                spl_r = CubicSpline(wqeh, vals.real, extrapolate=True)
                spl_i = CubicSpline(wqeh, vals.imag, extrapolate=True)
                dW_wab[:, a, b] = spl_r(w_grid) + 1j * spl_i(w_grid)

        return dW_wab

    def _calculate_sigma_mqeh(self, n_mG, deps_m, f_m, Wpm_Gpar,
                              Gpar_inverse_idx, N_c, pd0, phi_Gpar_az,
                              qplusGpar_abs, A):
        """Calculate self-energy contribution in the mQEH basis.

        For each G_parallel, project the pair density onto the mQEH
        density basis to get expansion coefficients, then contract
        with the Delta-W matrix.

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
        Gpar_inverse_idx : ndarray (nG,)
            Maps each G-vector index to its G_parallel group index.
        N_c : ndarray (3,)
            FFT grid dimensions.
        pd0 : SingleQPWDescriptor
            Plane-wave descriptor.
        phi_Gpar_az : ndarray (n_Gpar, nbasis, nz_qeh)
            Potential basis functions at each |q+G_parallel|.
        qplusGpar_abs : ndarray (n_Gpar,)
            |q + G_parallel| for each unique G_parallel.
        A : float
            In-plane unit cell area.
        """
        o_m = abs(deps_m)
        sgn_m = np.sign(deps_m + 1e-15)
        s_m = (1 + sgn_m * np.sign(0.5 - f_m)).astype(int) // 2

        nw = len(self.omega_w)
        nb = self.nbasis
        n_Gpar = Wpm_Gpar.shape[0]
        nG = n_mG.shape[1]

        beta = (2**0.5 - 1) * self.domega0 / self.omega2
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        o1_m = self.omega_w[w_m]
        o2_m = self.omega_w[w_m + 1]
        x = 1.0 / (self.qd.nbzkpts * 2 * pi * self.vol)

        # Compute mixed-space pair densities and project onto mQEH basis
        # For each G_parallel group, we need rho^n_m(G_parallel, z)
        # This requires inverse FFT along z of the pair densities
        # grouped by G_parallel.

        # Get the G-vector indices in the FFT grid
        Q_G = pd0.Q_qG[0]  # 1D indices into the 3D FFT grid
        i_cG = np.array(np.unravel_index(Q_G, N_c))  # 3D grid indices

        # Convert grid indices to actual G_z values
        B_cv = 2.0 * pi * self.gs.gd.icell_cv
        # G_z values for each G-vector
        Gz_idx = i_cG[2]  # z-component grid indices
        # Handle FFT index wrapping
        Gz_idx_wrapped = np.where(Gz_idx > N_c[2] // 2,
                                  Gz_idx - N_c[2], Gz_idx)

        # z-grid from the DFT cell
        Lz = abs(self.gs.gd.cell_cv[2, 2])
        dz_dft = Lz / N_c[2]
        z_dft = np.arange(N_c[2]) * dz_dft

        # For each G_parallel, do inverse FFT along z to get
        # the mixed-space pair density
        # Then project onto the mQEH potential basis to get C coefficients

        # Compute expansion coefficients C_{m,alpha}(G_par) for each band m
        # C_{m,alpha}(G_par) = int dz rho^n_m(G_par, z) phi^{l0}_alpha(z)
        # where phi is the potential basis function (dual to density basis)
        C_mGpar_a = np.zeros((len(deps_m), n_Gpar, nb), dtype=complex)

        for ig in range(n_Gpar):
            # Find all G-vectors belonging to this G_parallel group
            G_indices = np.where(Gpar_inverse_idx == ig)[0]
            if len(G_indices) == 0:
                continue

            # Get the Gz indices for this group
            gz_indices = Gz_idx_wrapped[G_indices]

            # For each band m, construct rho(G_par, z) via inverse FFT
            # n_mG[:, G_indices] are the Fourier coefficients for this G_par
            n_m_gz = n_mG[:, G_indices]  # (nbands, n_gz)

            # Do inverse FFT along z: accumulate exp(i Gz z) * n(Gz)
            # z-grid: use the QEH z-grid for projection
            z_qeh = self.z_z_qeh
            nz_qeh = len(z_qeh)

            # Compute exp(i Gz z) for all Gz and z points
            Gz_values = gz_indices * (2 * pi / Lz)
            # phase_zg: (nz_qeh, n_gz)
            phase_zg = np.exp(1j * np.outer(z_qeh, Gz_values))

            # rho_mz = n_m_gz @ phase_zg.T / Lz gives pair density
            # in mixed representation
            # Factor: inverse FFT normalization
            rho_mz = n_m_gz @ phase_zg.T / Lz  # (nbands, nz_qeh)

            # Project onto potential basis: C = int dz rho(z) phi(z)
            # phi_Gpar_az[ig] has shape (nbasis, nz_qeh)
            phi_az = phi_Gpar_az[ig]  # (nbasis, nz_qeh)
            C_mGpar_a[:, ig, :] = (
                rho_mz @ phi_az.conj().T * self.dz_qeh)

        # Now compute self-energy using the expansion coefficients
        # The factor 1/A appears because we sum over G_parallel:
        # (1/Omega) sum_G (...) = (1/A) sum_{G_par} (1/L) sum_{G_z} (...)
        # The (1/L) is absorbed into the inverse FFT normalization above,
        # and (1/Omega) is already in x, so we need an extra factor of
        # (Omega / A) = L to compensate for splitting 1/Omega into 1/(A*L).
        # Since dW was already multiplied by L, the net effect is that
        # x (which contains 1/Omega) handles the normalization correctly
        # when we sum over G_parallel contributions.
        # The pair density normalization: n_mG from GPAW includes dv,
        # so rho(G_par, z) = sum_{G_z} n(G_par,G_z) e^{iG_z z} / L_z
        # and C_alpha = int dz rho(G_par,z) phi_alpha(z) dz_qeh
        # The self-energy is:
        # (1/(N_q * 2pi)) * sum_{G_par} sum_{a,b} C*_a W_ab C_b / A
        # where the 1/A factor replaces 1/Omega from the parent.
        # Since x = 1/(N_q * 2pi * Omega) = 1/(N_q * 2pi * A * L),
        # and W already includes factor L, we get x*L = 1/(N_q*2pi*A).
        # So we should use x directly (no extra 1/A).
        sigma = 0.0
        dsigma = 0.0

        for o, o1, o2, sgn, s, w, C_Gpar_a in zip(
                o_m, o1_m, o2_m, sgn_m, s_m, w_m, C_mGpar_a):
            p = x * sgn
            sigma1 = 0.0
            sigma2 = 0.0

            for ig in range(n_Gpar):
                C_a = C_Gpar_a[ig]  # (nbasis,)
                W1_ab = Wpm_Gpar[ig, s * nw + w]      # (nbasis, nbasis)
                W2_ab = Wpm_Gpar[ig, s * nw + w + 1]  # (nbasis, nbasis)

                sigma1 += p * (C_a.conj() @ W1_ab @ C_a).imag
                sigma2 += p * (C_a.conj() @ W2_ab @ C_a).imag

            sigma += ((o - o1) * sigma2 + (o2 - o) * sigma1) / (o2 - o1)
            dsigma += sgn * (sigma2 - sigma1) / (o2 - o1)

        return sigma, dsigma
