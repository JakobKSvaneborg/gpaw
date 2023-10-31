from time import time, ctime
from datetime import timedelta

import numpy as np
from ase.units import Hartree, Bohr
from ase.dft import monkhorst_pack
from scipy.linalg import eigh

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.blacs import BlacsGrid, Redistributor
from gpaw.mpi import world, serial_comm
from gpaw.response import ResponseContext
from gpaw.response.df import write_response_function
from gpaw.response.coulomb_kernels import CoulombKernel
from gpaw.response.screened_interaction import initialize_w_calculator
from gpaw.response.paw import PWPAWCorrectionData
from gpaw.response.frequencies import FrequencyDescriptor
from gpaw.response.pair import KPointPairFactory, get_gs_and_context
from gpaw.response.pair_functions import SingleQPWDescriptor
from gpaw.response.chi0 import Chi0Calculator


class BSEBackend:
    def __init__(self, *, gs, context,
                 valence_bands, conduction_bands,
                 spinors=False,
                 ecut=10.,
                 scale=1.0,
                 nbands=None,
                 eshift=None,
                 gw_skn=None,
                 truncation=None,
                 integrate_gamma=1,
                 mode='BSE',
                 wfile=None,
                 write_h=False,
                 write_v=False):
        self.gs = gs
        self.context = context

        self.spinors = spinors
        self.scale = scale

        assert mode in ['RPA', 'TDHF', 'BSE']

        self.ecut = ecut / Hartree
        self.nbands = nbands
        self.mode = mode

        if integrate_gamma == 0 and truncation is not None:
            self.context.print('***WARNING*** Analytical Coulomb integration' +
                               ' is not expected to work with Coulomb ' +
                               'truncation. Use integrate_gamma=1')
        self.integrate_gamma = integrate_gamma
        self.wfile = wfile
        self.write_h = write_h
        self.write_v = write_v

        # Find q-vectors and weights in the IBZ:
        self.kd = self.gs.kd
        if -1 in self.kd.bz2bz_ks:
            self.context.print('***WARNING*** Symmetries may not be right. '
                               'Use gamma-centered grid to be sure')
        offset_c = 0.5 * ((self.kd.N_c + 1) % 2) / self.kd.N_c
        bzq_qc = monkhorst_pack(self.kd.N_c) + offset_c
        self.qd = KPointDescriptor(bzq_qc)
        self.qd.set_symmetry(self.gs.atoms, self.kd.symmetry)

        # bands
        self.spins = self.gs.nspins
        if self.spins == 2:
            if self.spinors:
                self.spinors = False
                self.context.print('***WARNING*** Presently the spinor ' +
                                   'version does not work for spin-polarized' +
                                   ' calculations. Performing scalar ' +
                                   'calculation')

        self.val_sn = self.parse_bands(valence_bands, band_type='valence')
        self.con_sn = self.parse_bands(conduction_bands,
                                       band_type='conduction')

        self.td = True
        for n in self.val_sn[0]:
            if n in self.con_sn[0]:
                self.td = False
        if len(self.val_sn) == 2:
            for n in self.val_sn[1]:
                if n in self.con_sn[1]:
                    self.td = False

        self.nv = len(self.val_sn[0])
        self.nc = len(self.con_sn[0])
        if eshift is not None:
            eshift /= Hartree
        if gw_skn is not None:
            assert self.nv + self.nc == len(gw_skn[0, 0])
            assert self.kd.nibzkpts == len(gw_skn[0])
            gw_skn = gw_skn[:, self.kd.bz2ibz_k]
            # assert self.kd.nbzkpts == len(gw_skn[0])
            gw_skn /= Hartree
        self.gw_skn = gw_skn
        self.eshift = eshift

        # Number of pair orbitals
        self.nS = self.kd.nbzkpts * self.nv * self.nc * self.spins
        self.nS *= (self.spinors + 1)**2

        self.coulomb = CoulombKernel.from_gs(self.gs, truncation=truncation)
        self.context.print(self.coulomb.description())

        self.print_initialization(self.td, self.eshift, self.gw_skn)

        # Chi0 object
        self._chi0calc = None  # Initialized later
        self._wcalc = None  # Initialized later

    @property
    def pair_calc(self):
        return self.kptpair_factory.pair_calculator()

    def parse_bands(self, bands, band_type='valence'):
        """Helper function that checks whether bands are correctly specified,
         and brings them to the format used later in the code.

        If the calculation is spin-polarized, band indices must
        be given explicitly as lists/arrays of shape (2,nbands) where the first
        index is for spin.

        If the calculation is not spin-polarized, either an integer (number of
        desired bands) or lists of band indices must be provided.

        band_type is an optional parameter that is only when a desired number
        of bands is given (rather than a list) to help figure out the correct
        band indices.
        """
        if hasattr(bands, '__iter__'):
            if self.spins == 2:
                if len(bands) != 2 or (len(bands[0]) != len(bands[1])):
                    raise ValueError('For a spin-polarized calculation, '
                                     'the same number of bands must be '
                                     'specified for each spin! valence and '
                                     'conduction bands must be lists of shape '
                                     '(2,n)')

            bands_sn = np.atleast_2d(bands)
            return bands_sn

        # if we get here, bands is not iterable
        # check that the specified input is valid

        if self.spins == 2:
            raise NotImplementedError('For a spin-polarized calculation, '
                                      'bands must be specified as lists '
                                      'of shape (2,n)')

        n_fully_occupied_bands, n_partially_occupied_bands = \
            self.gs.count_occupied_bands()

        if n_fully_occupied_bands != n_partially_occupied_bands:
            raise NotImplementedError('Automatic band generation is currently '
                                      'not implemented for metallic systems. '
                                      'Please specify band indices manually.')

        if band_type == 'valence':
            bands_sn = range(n_fully_occupied_bands - bands,
                             n_fully_occupied_bands)
        elif band_type == 'conduction':
            bands_sn = range(n_fully_occupied_bands,
                             n_fully_occupied_bands + bands)
        else:
            raise ValueError(f'Invalid band type: {band_type}')

        bands_sn = np.atleast_2d(bands_sn)
        return bands_sn

    def calculate(self, optical=True):

        if self.spinors:
            # Calculate spinors. Here m is index of eigenvalues with SOC
            # and n is the basis of eigenstates without SOC. Below m is used
            # for unoccupied states and n is used for occupied states so be
            # careful!

            self.context.print('Diagonalizing spin-orbit Hamiltonian')
            soc = self.gs.soc_eigenstates(scale=self.scale)
            e_mk = soc.eigenvalues().T
            v_kmn = soc.eigenvectors()
            e_mk /= Hartree

        # Parallelization stuff
        nK = self.kd.nbzkpts
        myKrange, myKsize, mySsize = self.parallelisation_sizes()

        # Calculate exchange interaction
        qpd0 = SingleQPWDescriptor.from_q(self.q_c, self.ecut, self.gs.gd)
        ikq_k = self.kd.find_k_plus_q(self.q_c)
        v_G = self.coulomb.V(qpd=qpd0, q_v=None)

        if optical:
            v_G[0] = 0.0

        self.kptpair_factory = KPointPairFactory(
            gs=self.gs,
            context=ResponseContext(txt='pair.txt', timer=None,
                                    comm=serial_comm))

        # Calculate direct (screened) interaction and PAW corrections
        if self.mode == 'RPA':
            pairden_paw_corr = self.gs.pair_density_paw_corrections
            pawcorr = pairden_paw_corr(qpd0)
        else:
            self.get_screened_potential()
            if (self.qd.ibzk_kc - self.q_c < 1.0e-6).all():
                iq0 = self.qd.bz2ibz_k[self.kd.where_is_q(self.q_c,
                                                          self.qd.bzk_kc)]
                pawcorr = self.pawcorr_q[iq0]  # Q_qaGii[iq0]
            else:
                pairden_paw_corr = self.gs.pair_density_paw_corrections
                pawcorr = pairden_paw_corr(qpd0)

        # Calculate pair densities, eigenvalues and occupations
        so = self.spinors + 1
        Nv, Nc = so * self.nv, so * self.nc
        Ns = self.spins
        rhoex_KsmnG = np.zeros((nK, Ns, Nv, Nc, len(v_G)), complex)
        # rhoG0_Ksmn = np.zeros((nK, Ns, Nv, Nc), complex)
        df_Ksmn = np.zeros((nK, Ns, Nv, Nc), float)  # -(ev - ec)
        deps_ksmn = np.zeros((myKsize, Ns, Nv, Nc), float)  # -(fv - fc)

        optical_limit = np.allclose(self.q_c, 0.0)

        get_pair = self.kptpair_factory.get_kpoint_pair
        get_pair_density = self.pair_calc.get_pair_density
        if self.spinors:
            # Get all pair densities to allow for SOC mixing
            # Use twice as many no-SOC states as BSE bands to allow mixing
            vi_s = [2 * self.val_sn[0, 0] - self.val_sn[0, -1] - 1]
            vf_s = [2 * self.con_sn[0, -1] - self.con_sn[0, 0] + 2]
            if vi_s[0] < 0:
                vi_s[0] = 0
            ci_s, cf_s = vi_s, vf_s
            ni, nf = vi_s[0], vf_s[0]
            mvi = 2 * self.val_sn[0, 0]
            mvf = 2 * (self.val_sn[0, -1] + 1)
            mci = 2 * self.con_sn[0, 0]
            mcf = 2 * (self.con_sn[0, -1] + 1)
        else:
            vi_s, vf_s = self.val_sn[:, 0], self.val_sn[:, -1] + 1
            ci_s, cf_s = self.con_sn[:, 0], self.con_sn[:, -1] + 1
        for ik, iK in enumerate(myKrange):
            for s in range(Ns):
                pair = get_pair(qpd0, s, iK,
                                vi_s[s], vf_s[s], ci_s[s], cf_s[s])
                m_m = np.arange(vi_s[s], vf_s[s])
                n_n = np.arange(ci_s[s], cf_s[s])
                if self.gw_skn is not None:
                    iKq = self.gs.kd.find_k_plus_q(self.q_c, [iK])[0]
                    epsv_m = self.gw_skn[s, iK, :self.nv]
                    epsc_n = self.gw_skn[s, iKq, self.nv:]
                    deps_ksmn[ik] = -(epsv_m[:, np.newaxis] - epsc_n)
                elif self.spinors:
                    iKq = self.gs.kd.find_k_plus_q(self.q_c, [iK])[0]
                    epsv_m = e_mk[mvi:mvf, iK]
                    epsc_n = e_mk[mci:mcf, iKq]
                    deps_ksmn[ik, s] = -(epsv_m[:, np.newaxis] - epsc_n)
                else:
                    deps_ksmn[ik, s] = -pair.get_transition_energies(m_m, n_n)

                df_mn = pair.get_occupation_differences(self.val_sn[s],
                                                        self.con_sn[s])
                rho_mnG = get_pair_density(qpd0, pair, m_m, n_n,
                                           pawcorr=pawcorr)
                if optical_limit:
                    n_mnv = self.pair_calc.get_optical_pair_density_head(
                        qpd0, pair, m_m, n_n)
                    rho_mnG[:, :, 0] = n_mnv[:, :, self.direction]
                if self.spinors:
                    v0_kmn = v_kmn[:, :, ::2]
                    v1_kmn = v_kmn[:, :, 1::2]
                    if optical_limit:
                        deps0_mn = -pair.get_transition_energies(m_m, n_n)
                        rho_mnG[:, :, 0] *= deps0_mn
                    df_Ksmn[iK, s, ::2, ::2] = df_mn
                    df_Ksmn[iK, s, ::2, 1::2] = df_mn
                    df_Ksmn[iK, s, 1::2, ::2] = df_mn
                    df_Ksmn[iK, s, 1::2, 1::2] = df_mn
                    vecv0_mn = v0_kmn[iK, mvi:mvf, ni:nf]
                    vecc0_mn = v0_kmn[iKq, mci:mcf, ni:nf]
                    rho_0mnG = np.dot(vecv0_mn.conj(),
                                      np.dot(vecc0_mn, rho_mnG))
                    vecv1_mn = v1_kmn[iK, mvi:mvf, ni:nf]
                    vecc1_mn = v1_kmn[iKq, mci:mcf, ni:nf]
                    rho_1mnG = np.dot(vecv1_mn.conj(),
                                      np.dot(vecc1_mn, rho_mnG))
                    rhoex_KsmnG[iK, s] = rho_0mnG + rho_1mnG
                    if optical_limit:
                        rhoex_KsmnG[iK, s, :, :, 0] /= deps_ksmn[ik, s]
                else:
                    df_Ksmn[iK, s] = pair.get_occupation_differences(m_m, n_n)
                    rhoex_KsmnG[iK, s] = rho_mnG

        if self.eshift is not None:
            deps_ksmn[np.where(df_Ksmn[myKrange] > 1.0e-3)] += self.eshift
            deps_ksmn[np.where(df_Ksmn[myKrange] < -1.0e-3)] -= self.eshift

        world.sum(df_Ksmn)
        world.sum(rhoex_KsmnG)

        self.rhoG0_S = np.reshape(rhoex_KsmnG[:, :, :, :, 0], -1)

        if hasattr(self, 'H_sS'):
            return

        # Calculate Hamiltonian
        t0 = time()
        self.context.print('Calculating {} matrix elements at q_c = {}'.format(
            self.mode, self.q_c))
        H_ksmnKsmn = np.zeros((myKsize, Ns, Nv, Nc, nK, Ns, Nv, Nc), complex)
        for ik1, iK1 in enumerate(myKrange):
            for s1 in range(Ns):
                kptv1 = self.kptpair_factory.get_k_point(
                    s1, iK1, vi_s[s1], vf_s[s1])
                kptc1 = self.kptpair_factory.get_k_point(
                    s1, ikq_k[iK1], ci_s[s1], cf_s[s1])
                rho1_mnG = rhoex_KsmnG[iK1, s1]

                # rhoG0_Ksmn[iK1, s1] = rho1_mnG[:, :, 0]
                rho1ccV_mnG = rho1_mnG.conj()[:, :] * v_G
                for s2 in range(Ns):
                    for Q_c in self.qd.bzk_kc:
                        iK2 = self.kd.find_k_plus_q(Q_c, [kptv1.K])[0]
                        rho2_mnG = rhoex_KsmnG[iK2, s2]

                        H_ksmnKsmn[ik1, s1, :, :, iK2, s2, :, :] += np.einsum(
                            'ijk,mnk->ijmn', rho1ccV_mnG, rho2_mnG,
                            optimize='optimal')

                        if not self.mode == 'RPA' and s1 == s2:
                            ikq = ikq_k[iK2]
                            kptv2 = self.kptpair_factory.get_k_point(
                                s1, iK2, vi_s[s1], vf_s[s1])
                            kptc2 = self.kptpair_factory.get_k_point(
                                s1, ikq, ci_s[s1], cf_s[s1])
                            rho3_mmG, iq = self.get_density_matrix(kptv1,
                                                                   kptv2)
                            rho4_nnG, iq = self.get_density_matrix(kptc1,
                                                                   kptc2)
                            if self.spinors:
                                vec0_mn = v0_kmn[iK1, mvi:mvf, ni:nf]
                                vec1_mn = v1_kmn[iK1, mvi:mvf, ni:nf]
                                vec2_mn = v0_kmn[iK2, mvi:mvf, ni:nf]
                                vec3_mn = v1_kmn[iK2, mvi:mvf, ni:nf]
                                rho_0mnG = np.dot(vec0_mn.conj(),
                                                  np.dot(vec2_mn, rho3_mmG))
                                rho_1mnG = np.dot(vec1_mn.conj(),
                                                  np.dot(vec3_mn, rho3_mmG))
                                rho3_mmG = rho_0mnG + rho_1mnG
                                vec0_mn = v0_kmn[ikq_k[iK1], mci:mcf, ni:nf]
                                vec1_mn = v1_kmn[ikq_k[iK1], mci:mcf, ni:nf]
                                vec2_mn = v0_kmn[ikq, mci:mcf, ni:nf]
                                vec3_mn = v1_kmn[ikq, mci:mcf, ni:nf]
                                rho_0mnG = np.dot(vec0_mn.conj(),
                                                  np.dot(vec2_mn, rho4_nnG))
                                rho_1mnG = np.dot(vec1_mn.conj(),
                                                  np.dot(vec3_mn, rho4_nnG))
                                rho4_nnG = rho_0mnG + rho_1mnG

                            W_mnmn = np.einsum('ijk,km,pqm->ipjq',
                                               rho3_mmG.conj(),
                                               self.W_qGG[iq],
                                               rho4_nnG,
                                               optimize='optimal')
                            W_mnmn *= Ns * so
                            H_ksmnKsmn[ik1, s1, :, :, iK2, s1] -= 0.5 * W_mnmn
            if iK1 % (myKsize // 5 + 1) == 0:
                dt = time() - t0
                tleft = dt * myKsize / (iK1 + 1) - dt
                self.context.print(
                    '  Finished %s pair orbitals in %s - Estimated %s left'
                    % ((iK1 + 1) * Nv * Nc * Ns * world.size, timedelta(
                        seconds=round(dt)), timedelta(seconds=round(tleft))))

        # if self.mode == 'BSE':
        #     del self.Q_qaGii, self.W_qGG, self.qpd_q

        H_ksmnKsmn /= self.gs.volume

        mySsize = myKsize * Nv * Nc * Ns
        if myKsize > 0:
            iS0 = myKrange[0] * Nv * Nc * Ns

        # world.sum(rhoG0_Ksmn)
        # self.rhoG0_S = np.reshape(rhoG0_Ksmn, -1)
        self.df_S = np.reshape(df_Ksmn, -1)
        if not self.td:
            self.excludef_S = np.where(np.abs(self.df_S) < 0.001)[0]
        # multiply by 2 when spin-paired and no SOC
        self.df_S *= 2.0 / nK / Ns / so
        self.deps_s = np.reshape(deps_ksmn, -1)
        H_sS = np.reshape(H_ksmnKsmn, (mySsize, self.nS))
        for iS in range(mySsize):
            # Multiply by occupations and adiabatic coupling
            H_sS[iS] *= self.df_S[iS0 + iS]
            # add bare transition energies
            H_sS[iS, iS0 + iS] += self.deps_s[iS]

        self.H_sS = H_sS

        if self.write_h:
            self.par_save('H_SS.ulm', 'H_SS', self.H_sS)

    def get_density_matrix(self, kpt1, kpt2):
        from gpaw.response.g0w0 import QSymmetryOp, get_nmG
        symop, iq = QSymmetryOp.get_symop_from_kpair(self.kd, self.qd,
                                                     kpt1, kpt2)
        qpd = self.qpd_q[iq]
        nG = qpd.ngmax
        pawcorr, I_G = symop.apply_symop_q(qpd, self.pawcorr_q[iq], kpt1, kpt2)

        rho_mnG = np.zeros((len(kpt1.eps_n), len(kpt2.eps_n), nG),
                           complex)
        for m in range(len(rho_mnG)):
            rho_mnG[m] = get_nmG(kpt1, kpt2, pawcorr, m, qpd, I_G,
                                 self.pair_calc)
        return rho_mnG, iq

    def get_screened_potential(self):

        if hasattr(self, 'W_qGG'):
            return

        if self.wfile is not None:
            # Read screened potential from file
            try:
                data = np.load(self.wfile + '.npz')
                self.qpd_q = data['pd']
                assert len(data['pd']) == len(data['Q'])
                self.pawcorr_q = [
                    PWPAWCorrectionData(
                        Q_aGii, qpd=qpd,
                        pawdatasets=self.gs.pawdataset_by_species,
                        pos_av=self.gs.get_pos_av(),
                        atomrotations=self.gs.atomrotations)
                    for Q_aGii, qpd in zip(data['Q'], self.qpd_q)]
                self.W_qGG = data['W']
                self.context.print('Reading screened potential from % s' %
                                   self.wfile)
            except FileNotFoundError:
                self.calculate_screened_potential()
                self.context.print('Saving screened potential to % s' %
                                   self.wfile)
                if world.rank == 0:
                    np.savez(self.wfile,
                             Q=[pawcorr.Q_aGii for pawcorr in self.pawcorr_q],
                             pd=self.qpd_q, W=self.W_qGG)
        else:
            self.calculate_screened_potential()

    def initialize_chi0_calculator(self):
        """Initialize the Chi0 object to compute the static
        susceptibility."""

        wd = FrequencyDescriptor([0.0])
        kptpair_factory = KPointPairFactory(
            gs=self.gs,
            context=self.context.with_txt('chi0.txt'))

        self._chi0calc = Chi0Calculator(
            wd=wd,
            kptpair_factory=kptpair_factory,
            eta=0.001,
            ecut=self.ecut * Hartree,
            intraband=False,
            hilbert=False,
            nbands=self.nbands)

        self.blockcomm = self._chi0calc.chi0_body_calc.integrator.blockcomm

    def calculate_screened_potential(self):
        """Calculate W_GG(q)"""

        self.pawcorr_q = []
        self.W_qGG = []
        self.qpd_q = []

        # F.N: Moved this here. chi0 will be calculated by WCalculator
        if self._chi0calc is None:
            self.initialize_chi0_calculator()
        if self._wcalc is None:
            wcontext = ResponseContext(txt='w.txt', comm=world)
            self._wcalc = initialize_w_calculator(
                self._chi0calc, wcontext,
                coulomb=self.coulomb,
                integrate_gamma=self.integrate_gamma)
        t0 = time()
        self.context.print('Calculating screened potential')
        for iq, q_c in enumerate(self.qd.ibzk_kc):
            chi0 = self._chi0calc.calculate(q_c)
            W_wGG = self._wcalc.calculate_W_wGG(chi0)
            W_GG = W_wGG[0]
            # This is such a terrible way to access the paw
            # corrections. Attributes should not be groped like
            # this... Change in the future! XXX
            self.pawcorr_q.append(self._chi0calc.chi0_body_calc.pawcorr)
            self.qpd_q.append(chi0.qpd)
            self.W_qGG.append(W_GG)

            if iq % (self.qd.nibzkpts // 5 + 1) == 2:
                dt = time() - t0
                tleft = dt * self.qd.nibzkpts / (iq + 1) - dt
                self.context.print(
                    '  Finished {} q-points in {} - Estimated {} left'.format(
                        iq + 1, timedelta(seconds=round(dt)), timedelta(
                            seconds=round(tleft))))

    def diagonalize(self):

        self.context.print('Diagonalizing Hamiltonian')
        """The t and T represent local and global
           eigenstates indices respectively
        """

        # Non-Hermitian matrix can only use linalg.eig
        if not self.td:
            self.context.print('  Using numpy.linalg.eig...')
            self.context.print('  Eliminated %s pair orbitals' % len(
                self.excludef_S))

            self.H_SS = self.collect_A_SS(self.H_sS)
            self.w_T = np.zeros(self.nS - len(self.excludef_S), complex)
            if world.rank == 0:
                self.H_SS = np.delete(self.H_SS, self.excludef_S, axis=0)
                self.H_SS = np.delete(self.H_SS, self.excludef_S, axis=1)
                self.w_T, self.v_ST = np.linalg.eig(self.H_SS)
            world.broadcast(self.w_T, 0)
            self.df_S = np.delete(self.df_S, self.excludef_S)
            self.rhoG0_S = np.delete(self.rhoG0_S, self.excludef_S)
        # Here the eigenvectors are returned as complex conjugated rows
        else:
            if world.size == 1:
                self.context.print('  Using lapack...')
                self.w_T, self.v_St = eigh(self.H_sS)
            else:
                self.context.print('  Using scalapack...')
                nS = self.nS
                ns = -(-self.kd.nbzkpts // world.size) * (
                    self.nv * self.nc *
                    self.spins *
                    (self.spinors + 1)**2)
                grid = BlacsGrid(world, world.size, 1)
                desc = grid.new_descriptor(nS, nS, ns, nS)

                desc2 = grid.new_descriptor(nS, nS, 2, 2)
                H_tmp = desc2.zeros(dtype=complex)
                r = Redistributor(world, desc, desc2)
                r.redistribute(self.H_sS, H_tmp)

                self.w_T = np.empty(nS)
                v_tmp = desc2.empty(dtype=complex)
                desc2.diagonalize_dc(H_tmp, v_tmp, self.w_T)

                r = Redistributor(grid.comm, desc2, desc)
                self.v_St = desc.zeros(dtype=complex)
                r.redistribute(v_tmp, self.v_St)
                self.v_St = self.v_St.conj().T

        if self.write_v and self.td:
            # Cannot use par_save without td
            self.par_save('v_TS.ulm', 'v_TS', self.v_St.T)

        return

    def get_bse_matrix(self, q_c=[0.0, 0.0, 0.0], direction=0,
                       readfile=None, optical=True):
        """Calculate and diagonalize BSE matrix"""

        self.q_c = q_c
        self.direction = direction

        if readfile is None:
            self.calculate(optical=optical)
            if hasattr(self, 'w_T'):
                return
            self.diagonalize()
        elif readfile == 'H_SS':
            self.context.print('Reading Hamiltonian from file')
            self.par_load('H_SS.ulm', 'H_SS')
            self.diagonalize()
        elif readfile == 'v_TS':
            self.context.print('Reading eigenstates from file')
            self.par_load('v_TS.ulm', 'v_TS')
        else:
            raise ValueError('%s array not recognized' % readfile)

        return

    def get_vchi(self, w_w=None, eta=0.1, q_c=[0.0, 0.0, 0.0],
                 direction=0, readfile=None, optical=True,
                 write_eig=None):
        """Returns v * chi where v is the bare Coulomb interaction"""

        self.get_bse_matrix(q_c=q_c, direction=direction,
                            readfile=readfile, optical=optical)

        w_T = self.w_T
        rhoG0_S = self.rhoG0_S
        df_S = self.df_S

        self.context.print('Calculating response function at %s frequency '
                           'points' % len(w_w))
        vchi_w = np.zeros(len(w_w), dtype=complex)

        if not self.td:
            C_T = np.zeros(self.nS - len(self.excludef_S), complex)
            if world.rank == 0:
                A_T = np.dot(rhoG0_S, self.v_ST)
                B_T = np.dot(rhoG0_S * df_S, self.v_ST)
                tmp = np.dot(self.v_ST.conj().T, self.v_ST)
                overlap_tt = np.linalg.inv(tmp)
                C_T = np.dot(B_T.conj(), overlap_tt.T) * A_T
            world.broadcast(C_T, 0)
        else:
            A_t = np.dot(rhoG0_S, self.v_St)
            B_t = np.dot(rhoG0_S * df_S, self.v_St)
            if world.size == 1:
                C_T = B_t.conj() * A_t
            else:
                Nv = self.nv * (self.spinors + 1)
                Nc = self.nc * (self.spinors + 1)
                Ns = self.spins
                nS = self.nS
                ns = -(-self.kd.nbzkpts // world.size) * Nv * Nc * Ns
                grid = BlacsGrid(world, world.size, 1)
                desc = grid.new_descriptor(nS, 1, ns, 1)
                C_t = desc.empty(dtype=complex)
                C_t[:, 0] = B_t.conj() * A_t
                C_T = desc.collect_on_master(C_t)[:, 0]
                if world.rank != 0:
                    C_T = np.empty(nS, dtype=complex)
                world.broadcast(C_T, 0)

        eta /= Hartree
        for iw, w in enumerate(w_w / Hartree):
            tmp_T = 1. / (w - w_T + 1j * eta)
            vchi_w[iw] += np.dot(tmp_T, C_T)
        vchi_w *= 4 * np.pi / self.gs.volume

        if not np.allclose(self.q_c, 0.0):
            cell_cv = self.gs.gd.cell_cv
            B_cv = 2 * np.pi * np.linalg.inv(cell_cv).T
            q_v = np.dot(q_c, B_cv)
            vchi_w /= np.dot(q_v, q_v)

        """Check f-sum rule."""
        nv = self.gs.nvalence
        dw_w = (w_w[1:] - w_w[:-1]) / Hartree
        wchi_w = (w_w[1:] * vchi_w[1:] + w_w[:-1] * vchi_w[:-1]) / Hartree / 2
        N = -np.dot(dw_w, wchi_w.imag) * self.gs.volume / (2 * np.pi**2)
        self.context.print('', flush=False)
        self.context.print('Checking f-sum rule:', flush=False)
        self.context.print(f'  Valence = {nv}, N = {N:f}', flush=False)
        self.context.print('')

        if write_eig is not None:
            assert isinstance(write_eig, str)
            filename = write_eig
            if world.rank == 0:
                write_bse_eigenvalues(filename, self.mode,
                                      self.w_T * Hartree, C_T)

        return vchi_w

    def get_dielectric_function(self, w_w=None, eta=0.1,
                                q_c=[0.0, 0.0, 0.0], direction=0,
                                filename='df_bse.csv', readfile=None,
                                write_eig='eig.dat'):
        """Returns and writes real and imaginary part of the dielectric
        function.

        w_w: list of frequencies (eV)
            Dielectric function is calculated at these frequencies
        eta: float
            Lorentzian broadening of the spectrum (eV)
        q_c: list of three floats
            Wavevector in reduced units on which the response is calculated
        direction: int
            if q_c = [0, 0, 0] this gives the direction in cartesian
            coordinates - 0=x, 1=y, 2=z
        filename: str
            data file on which frequencies, real and imaginary part of
            dielectric function is written
        readfile: str
            If H_SS is given, the method will load the BSE Hamiltonian
            from H_SS.ulm. If v_TS is given, the method will load the
            eigenstates from v_TS.ulm
        write_eig: str
            File on which the BSE eigenvalues are written
        """

        epsilon_w = -self.get_vchi(w_w=w_w, eta=eta, q_c=q_c,
                                   direction=direction,
                                   readfile=readfile, optical=True,
                                   write_eig=write_eig)
        epsilon_w += 1.0

        if world.rank == 0 and filename is not None:
            write_response_function(filename, w_w,
                                    epsilon_w.real, epsilon_w.imag)
        world.barrier()

        self.context.print('Calculation completed at:', ctime(), flush=False)
        self.context.print('')

        return w_w, epsilon_w

    def get_eels_spectrum(self, w_w=None, eta=0.1,
                          q_c=[0.0, 0.0, 0.0], direction=0,
                          filename='df_bse.csv', readfile=None,
                          write_eig='eig.dat'):
        """Returns and writes real and imaginary part of the dielectric
        function.

        w_w: list of frequencies (eV)
            Dielectric function is calculated at these frequencies
        eta: float
            Lorentzian broadening of the spectrum (eV)
        q_c: list of three floats
            Wavevector in reduced units on which the response is calculated
        direction: int
            if q_c = [0, 0, 0] this gives the direction in cartesian
            coordinates - 0=x, 1=y, 2=z
        filename: str
            data file on which frequencies, real and imaginary part of
            dielectric function is written
        readfile: str
            If H_SS is given, the method will load the BSE Hamiltonian
            from H_SS.ulm. If v_TS is given, the method will load the
            eigenstates from v_TS.ulm
        write_eig: str
            File on which the BSE eigenvalues are written
        """

        eels_w = -self.get_vchi(w_w=w_w, eta=eta, q_c=q_c, direction=direction,
                                readfile=readfile, optical=False,
                                write_eig=write_eig).imag

        if world.rank == 0 and filename is not None:
            write_spectrum(filename, w_w, eels_w)
        world.barrier()

        self.context.print('Calculation completed at:', ctime(), flush=False)
        self.context.print('')

        return w_w, eels_w

    def get_polarizability(self, w_w=None, eta=0.1,
                           q_c=[0.0, 0.0, 0.0], direction=0,
                           filename='pol_bse.csv', readfile=None,
                           write_eig='eig.dat'):
        r"""Calculate the polarizability alpha.
        In 3D the imaginary part of the polarizability is related to the
        dielectric function by Im(eps_M) = 4 pi * Im(alpha). In systems
        with reduced dimensionality the converged value of alpha is
        independent of the cell volume. This is not the case for eps_M,
        which is ill defined. A truncated Coulomb kernel will always give
        eps_M = 1.0, whereas the polarizability maintains its structure.
        pbs should be a list of booleans giving the periodic directions.

        By default, generate a file 'pol_bse.csv'. The three colomns are:
        frequency (eV), Real(alpha), Imag(alpha). The dimension of alpha
        is \AA to the power of non-periodic directions.
        """

        pbc_c = self.gs.pbc

        V = self.gs.nonpbc_cell_product()

        optical = (self.coulomb.truncation is None)

        vchi_w = self.get_vchi(w_w=w_w, eta=eta, q_c=q_c, direction=direction,
                               readfile=readfile, optical=optical,
                               write_eig=write_eig)
        alpha_w = -V * vchi_w / (4 * np.pi)
        alpha_w *= Bohr**(sum(~pbc_c))

        if world.rank == 0 and filename is not None:
            write_response_function(filename, w_w, alpha_w.real, alpha_w.imag)

        self.context.print('Calculation completed at:', ctime(), flush=False)
        self.context.print('')

        return w_w, alpha_w

    def par_save(self, filename, name, A_sS):
        import ase.io.ulm as ulm

        if world.size == 1:
            A_XS = A_sS
        else:
            A_XS = self.collect_A_SS(A_sS)

        if world.rank == 0:
            w = ulm.open(filename, 'w')
            if name == 'v_TS':
                w.write(w_T=self.w_T)
            # w.write(nS=self.nS)
            w.write(rhoG0_S=self.rhoG0_S)
            w.write(df_S=self.df_S)
            w.write(A_XS=A_XS)
            w.close()
        world.barrier()

    def par_load(self, filename, name):
        import ase.io.ulm as ulm

        if world.rank == 0:
            r = ulm.open(filename, 'r')
            if name == 'v_TS':
                self.w_T = r.w_T
            self.rhoG0_S = r.rhoG0_S
            self.df_S = r.df_S
            A_XS = r.A_XS
            r.close()
        else:
            if name == 'v_TS':
                self.w_T = np.zeros((self.nS), dtype=float)
            self.rhoG0_S = np.zeros((self.nS), dtype=complex)
            self.df_S = np.zeros((self.nS), dtype=float)
            A_XS = None

        world.broadcast(self.rhoG0_S, 0)
        world.broadcast(self.df_S, 0)

        if name == 'H_SS':
            self.H_sS = self.distribute_A_SS(A_XS)

        if name == 'v_TS':
            world.broadcast(self.w_T, 0)
            self.v_St = self.distribute_A_SS(A_XS, transpose=True)

    def collect_A_SS(self, A_sS):
        if world.rank == 0:
            A_SS = np.zeros((self.nS, self.nS), dtype=complex)
            A_SS[:len(A_sS)] = A_sS
            Ntot = len(A_sS)
            for rank in range(1, world.size):
                nkr, nk, ns = self.parallelisation_sizes(rank)
                buf = np.empty((ns, self.nS), dtype=complex)
                world.receive(buf, rank, tag=123)
                A_SS[Ntot:Ntot + ns] = buf
                Ntot += ns
        else:
            world.send(A_sS, 0, tag=123)
        world.barrier()
        if world.rank == 0:
            return A_SS

    def distribute_A_SS(self, A_SS, transpose=False):
        if world.rank == 0:
            for rank in range(0, world.size):
                nkr, nk, ns = self.parallelisation_sizes(rank)
                if rank == 0:
                    A_sS = A_SS[0:ns]
                    Ntot = ns
                else:
                    world.send(A_SS[Ntot:Ntot + ns], rank, tag=123)
                    Ntot += ns
        else:
            nkr, nk, ns = self.parallelisation_sizes()
            A_sS = np.empty((ns, self.nS), dtype=complex)
            world.receive(A_sS, 0, tag=123)
        world.barrier()
        if transpose:
            A_sS = A_sS.T
        return A_sS

    def parallelisation_sizes(self, rank=None):
        if rank is None:
            rank = world.rank
        nK = self.kd.nbzkpts
        myKsize = -(-nK // world.size)
        myKrange = range(rank * myKsize,
                         min((rank + 1) * myKsize, nK))
        myKsize = len(myKrange)
        mySsize = myKsize * self.nv * self.nc * self.spins
        mySsize *= (1 + self.spinors)**2
        return myKrange, myKsize, mySsize

    def print_initialization(self, td, eshift, gw_skn):
        isl = ['----------------------------------------------------------',
               f'{self.mode} Hamiltonian',
               '----------------------------------------------------------',
               f'Started at:  {ctime()}', '',
               'Atoms                          : '
               f'{self.gs.atoms.get_chemical_formula(mode="hill")}',
               f'Ground state XC functional     : {self.gs.xcname}',
               f'Valence electrons              : {self.gs.nvalence}',
               f'Spinor calculations            : {self.spinors}',
               f'Number of bands                : {self.gs.bd.nbands}',
               f'Number of spins                : {self.gs.nspins}',
               f'Number of k-points             : {self.kd.nbzkpts}',
               f'Number of irreducible k-points : {self.kd.nibzkpts}',
               f'Number of q-points             : {self.qd.nbzkpts}',
               f'Number of irreducible q-points : {self.qd.nibzkpts}', '']

        for q in self.qd.ibzk_kc:
            isl.append(f'    q: [{q[0]:1.4f} {q[1]:1.4f} {q[2]:1.4f}]')
        isl.append('')
        if gw_skn is not None:
            isl.append('User specified BSE bands')
        isl.extend([f'Response PW cutoff             : {self.ecut * Hartree} '
                    f'eV',
                    f'Screening bands included       : {self.nbands}'])
        if len(self.val_sn) == 1:
            isl.extend([f'Valence bands                  : {self.val_sn[0]}',
                        f'Conduction bands               : {self.con_sn[0]}'])
        else:
            isl.extend([f'Valence bands                  : {self.val_sn[0]}'
                        f' {self.val_sn[1]}',
                        f'Conduction bands               : {self.con_sn[0]}'
                        f' {self.con_sn[1]}'])
        if eshift is not None:
            isl.append(f'Scissors operator              : {eshift * Hartree}'
                       f'eV')
        isl.extend([
            f'Tamm-Dancoff approximation     : {td}',
            f'Number of pair orbitals        : {self.nS}',
            '',
            f'Truncation of Coulomb kernel   : {self.coulomb.truncation}'])
        if self.integrate_gamma == 0:
            isl.append(
                'Coulomb integration scheme     : Analytical - gamma only')
        elif self.integrate_gamma == 1:
            isl.append(
                'Coulomb integration scheme     : Numerical - all q-points')
        else:
            pass
        isl.extend([
            '',
            '----------------------------------------------------------',
            '----------------------------------------------------------',
            '',
            f'Parallelization - Total number of CPUs   : {world.size}',
            '  Screened potential',
            f'    K-point/band decomposition           : {world.size}',
            '  Hamiltonian',
            f'    Pair orbital decomposition           : {world.size}'])
        self.context.print('\n'.join(isl))


class BSE(BSEBackend):
    def __init__(self, calc=None, txt='-', **kwargs):
        """Creates the BSE object

        calc: str or calculator object
            The string should refer to the .gpw file contaning KS orbitals
        ecut: float
            Plane wave cutoff energy (eV)
        nbands: int
            Number of bands used for the screened interaction
        valence_bands: list
            Valence bands used in the BSE Hamiltonian
        conduction_bands: list
            Conduction bands used in the BSE Hamiltonian
        eshift: float
            Scissors operator opening the gap (eV)
        gw_skn: list / array
            List or array defining the gw quasiparticle energies used in
            the BSE Hamiltonian. Should match spin, k-points and
            valence/conduction bands
        truncation: str or None
            Coulomb truncation scheme. Can be None or 2D.
        integrate_gamma: int
            Method to integrate the Coulomb interaction. 1 is a numerical
            integration at all q-points with G=[0,0,0] - this breaks the
            symmetry slightly. 0 is analytical integration at q=[0,0,0] only -
            this conserves the symmetry. integrate_gamma=2 is the same as 1,
            but the average is only carried out in the non-periodic directions.
        txt: str
            txt output
        mode: str
            Theory level used. can be RPA TDHF or BSE. Only BSE is screened.
        wfile: str
            File for saving screened interaction and some other stuff
            needed later
        write_h: bool
            If True, write the BSE Hamiltonian to H_SS.ulm.
        write_v: bool
            If True, write eigenvalues and eigenstates to v_TS.ulm
        """
        gs, context = get_gs_and_context(
            calc, txt, world=world, timer=None)

        super().__init__(gs=gs, context=context, **kwargs)


def write_bse_eigenvalues(filename, mode, w_w, C_w):
    with open(filename, 'w') as fd:
        print('# %s eigenvalues (in eV) and weights' % mode, file=fd)
        print('# Number   eig   weight', file=fd)
        for iw, (w, C) in enumerate(zip(w_w, C_w)):
            print('%8d %12.6f %12.16f' % (iw, w.real, C.real),
                  file=fd)


def read_bse_eigenvalues(filename):
    _, w_w, C_w = np.loadtxt(filename, unpack=True)
    return w_w, C_w


def write_spectrum(filename, w_w, A_w):
    with open(filename, 'w') as fd:
        for w, A in zip(w_w, A_w):
            print(f'{w:.9f}, {A:.9f}', file=fd)


def read_spectrum(filename):
    w_w, A_w = np.loadtxt(filename, delimiter=',',
                          unpack=True)
    return w_w, A_w
