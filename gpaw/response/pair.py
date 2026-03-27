import numpy as np

from gpaw.response import ResponseContext, ResponseGroundStateAdapter, timer
from gpaw.response.pw_parallelization import block_partition
from gpaw.utilities.blas import mmm


def _fine_to_coarse_3d(i_cG, N_c, M_c):
    """Map 3D FFT grid indices from a fine grid to a coarse grid.

    Correctly handles negative G-vectors (indices >= N/2) by converting
    to signed crystal coordinates before taking the modulus.
    """
    N_c = np.asarray(N_c)
    M_c = np.asarray(M_c)
    # Convert fine-grid indices to signed G-vector crystal coordinates
    G_cG = np.where(i_cG >= N_c[:, np.newaxis] // 2,
                    i_cG - N_c[:, np.newaxis],
                    i_cG)
    # Map to coarse grid (Python % always returns non-negative)
    return G_cG % M_c[:, np.newaxis]


class KPoint:
    def __init__(self, s, K, n1, n2, blocksize, na, nb,
                 ut_nR, eps_n, f_n, P_ani, k_c):
        self.s = s    # spin index
        self.K = K    # BZ k-point index
        self.n1 = n1  # first band
        self.n2 = n2  # first band not included
        self.blocksize = blocksize
        self.na = na  # first band of block
        self.nb = nb  # first band of block not included
        self.ut_nR = ut_nR      # periodic part of wave functions in real-space
        self.eps_n = eps_n      # eigenvalues
        self.f_n = f_n          # occupation numbers
        self.P_ani = P_ani      # PAW projections
        self.k_c = k_c  # k-point coordinates


class KPointPair:
    """This class defines the kpoint-pair container object.

    Used for calculating pair quantities it contains two kpoints,
    and an associated set of Fourier components."""
    def __init__(self, kpt1, kpt2, Q_G):
        self.kpt1 = kpt1
        self.kpt2 = kpt2
        self.Q_G = Q_G

    def get_transition_energies(self):
        """Return the energy difference for specified bands."""
        kpt1 = self.kpt1
        kpt2 = self.kpt2
        deps_nm = kpt1.eps_n[:, np.newaxis] - kpt2.eps_n
        return deps_nm

    def get_occupation_differences(self):
        """Get difference in occupation factor between specified bands."""
        kpt1 = self.kpt1
        kpt2 = self.kpt2
        df_nm = kpt1.f_n[:, np.newaxis] - kpt2.f_n
        return df_nm


class KPointPairFactory:
    def __init__(self, gs, context):
        self.gs = gs
        self.context = context
        assert self.gs.kd.symmetry.symmorphic
        assert self.gs.world.size == 1
        # Cache for IBZ iFFT results.  Multiple BZ k-points K map to the
        # same IBZ k-point ik, so pd.ifft(psit_nG[n], ik) gives identical
        # results for all of them.  Only map_pseudo_wave (a cheap grid
        # permutation) differs per K.
        #
        # To avoid storing iFFTs for the entire IBZ (which would consume
        # too much memory), the cache holds results for at most one
        # (s, ik) pair at a time.  When a different IBZ k-point is
        # requested, the cache is evicted.  This still helps because the
        # integrator visits many BZ k-points that share the same ik in
        # succession (K1 and K2 in a pair, and consecutive domain points
        # that map to the same ik).
        self._ibz_ifft_cache = {}
        self._ibz_ifft_cache_key = None  # current (s, ik)

    @timer('Get a k-point')
    def get_k_point(self, s, K, n1, n2, blockcomm=None, pw_mode=False):
        """Return wave functions for a specific k-point and spin.

        s: int
            Spin index (0 or 1).
        K: int
            BZ k-point index.
        n1, n2: int
            Range of bands to include.
        pw_mode: bool
            If True, skip the expensive inverse FFT to real space.
            The resulting KPoint will have ut_nR=None, suitable only
            for the coarse-grid pair density method.
        """

        assert n1 <= n2

        gs = self.gs
        kd = gs.kd

        if blockcomm:
            nblocks = blockcomm.size
            rank = blockcomm.rank
        else:
            nblocks = 1
            rank = 0

        blocksize = (n2 - n1 + nblocks - 1) // nblocks
        na = min(n1 + rank * blocksize, n2)
        nb = min(na + blocksize, n2)

        ik = kd.bz2ibz_k[K]
        assert kd.comm.size == 1
        kpt = gs.kpt_ks[ik][s]

        assert n2 <= len(kpt.eps_n), \
            'Increase GS-nbands or decrease chi0-nbands!'
        eps_n = kpt.eps_n[n1:n2]
        f_n = kpt.f_n[n1:n2] / kpt.weight

        k_c = self.gs.ibz2bz[K].map_kpoint()

        if not pw_mode:
            with self.context.timer('load wfs'):
                psit_nG = kpt.psit_nG

                # Evict cache if the IBZ k-point changed, so we only
                # ever store iFFTs for one (s, ik) at a time.
                sik = (s, ik)
                if sik != self._ibz_ifft_cache_key:
                    self._ibz_ifft_cache.clear()
                    self._ibz_ifft_cache_key = sik

                ut_nR = gs.gd.empty(nb - na, gs.dtype)
                for n in range(na, nb):
                    # pd.ifft(psit_nG[n], ik) depends only on (s, ik, n),
                    # not on the BZ k-point K.  The symmetry rotation
                    # (map_pseudo_wave) is cheap and K-dependent.
                    ut_ibz_R = self._ibz_ifft_cache.get(n)
                    if ut_ibz_R is None:
                        ut_ibz_R = gs.pd.ifft(psit_nG[n], ik)
                        self._ibz_ifft_cache[n] = ut_ibz_R
                    ut_nR[n - na] = self.gs.ibz2bz[K].map_pseudo_wave(
                        ut_ibz_R)
        else:
            ut_nR = None

        with self.context.timer('Load projections'):
            if nb - na > 0:
                proj = kpt.projections.new(nbands=nb - na, bcomm=None)
                proj.array[:] = kpt.projections.array[na:nb]
                proj = self.gs.ibz2bz[K].map_projections(proj)
                P_ani = [P_ni for _, P_ni in proj.items()]
            else:
                P_ani = []

        return KPoint(s, K, n1, n2, blocksize, na, nb,
                      ut_nR, eps_n, f_n, P_ani, k_c)

    @timer('Get kpoint pair')
    def get_kpoint_pair(self, qpd, s, K, n1, n2, m1, m2,
                        blockcomm=None, flipspin=False, pw_mode=False):
        assert m1 <= m2
        assert n1 <= n2

        kptfinder = self.gs.kpoints.kptfinder

        k_c = self.gs.kd.bzk_kc[K]
        K1 = kptfinder.find(k_c)
        K2 = kptfinder.find(k_c + qpd.q_c)
        s1 = s
        s2 = (s + flipspin) % 2

        with self.context.timer('get k-points'):
            kpt1 = self.get_k_point(s1, K1, n1, n2, pw_mode=pw_mode)
            kpt2 = self.get_k_point(s2, K2, m1, m2, blockcomm=blockcomm,
                                    pw_mode=pw_mode)

        with self.context.timer('fft indices'):
            Q_G = phase_shifted_fft_indices(kpt1.k_c, kpt2.k_c, qpd)

        return KPointPair(kpt1, kpt2, Q_G)

    def pair_calculator(self, blockcomm=None):
        # We have decoupled the actual pair density calculator
        # from the kpoint factory, but it's still handy to
        # keep this shortcut -- for now.
        if blockcomm is None:
            blockcomm, _ = block_partition(self.context.comm, nblocks=1)
        return ActualPairDensityCalculator(self, blockcomm)


class ActualPairDensityCalculator:
    def __init__(self, kptpair_factory, blockcomm):
        # it seems weird to use kptpair_factory only for this
        self.gs = kptpair_factory.gs
        self.context = kptpair_factory.context
        self.blockcomm = blockcomm
        self.ut_sKnvR = None  # gradient of wave functions for optical limit

    def get_optical_pair_density(self, qpd, kptpair, n_n, m_m, *,
                                 pawcorr, block=False):
        """Get the full optical pair density, including the optical limit head
        for q=0."""

        nG = qpd.ngmax
        # P = (x, y, z, G1, G2, ...)
        n_nmP = np.empty((len(n_n), len(m_m), nG + 2), dtype=qpd.dtype)
        tmp_nmG = n_nmP[:, :, 2:]
        self.get_pair_density(qpd, kptpair, n_n, m_m,
                              pawcorr=pawcorr, block=block,
                              output_buffer=tmp_nmG)
        n_nmv = n_nmP[:, :, :3]
        self.get_optical_pair_density_head(qpd, kptpair, n_n, m_m,
                                           block=block,
                                           output_buffer=n_nmv)
        return n_nmP

    @timer('get_pair_density')
    def get_pair_density(self, qpd, kptpair, n_n, m_m, *,
                         pawcorr, block=False, output_buffer=None):
        """Get pair density for a kpoint pair."""
        cpd = self.calculate_pair_density

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        Q_G = kptpair.Q_G  # Fourier components of kpoint pair
        nG = len(Q_G)
        if output_buffer is None:
            n_nmG = np.zeros((len(n_n), len(m_m), nG), qpd.dtype)
        else:
            assert output_buffer.shape == (len(n_n), len(m_m), nG)
            assert output_buffer.dtype == qpd.dtype
            n_nmG = output_buffer
        for j, n in enumerate(n_n):
            Q_G = kptpair.Q_G
            with self.context.timer('conj'):
                ut1cc_R = kpt1.ut_nR[n - kpt1.na].conj()
            with self.context.timer('paw'):
                C1_aGi = pawcorr.multiply(kpt1.P_ani, band=n - kpt1.na)
                n_nmG[j] = cpd(ut1cc_R, C1_aGi, kpt2, qpd, Q_G, block=block)

        return n_nmG

    @timer('get_pair_density_coarsegrid')
    def get_pair_density_coarsegrid(self, qpd, kptpair, n_n, m_m, *,
                                    pawcorr, block=False, output_buffer=None,
                                    ecut_pair):
        r"""Get pair density using coarse-grid FFT with truncated PW expansion.

        The exact pair density is the convolution:

            ρ(G_j+q) = (Ω/N²) Σ_{G' ∈ S_gs} c*_n(G') c_m(G'+G_j+shift)

        where S_gs = {G : ½|G+k|² < ecut_gs}. This method approximates it by
        truncating the sum to a smaller set:

            ρ̃(G_j+q) = (Ω/N²) Σ_{G' ∈ S_pair} c*_n(G') c_m(G'+G_j+shift)

        where S_pair = {G : ½|G+k|² < ecut_pair} ⊂ S_gs. The truncation
        discards contributions from high-frequency PW coefficients, which are
        small for well-converged ground states.

        The truncated convolution is computed via FFT on a coarser real-space
        grid whose dimensions are set by ecut_pair rather than ecut_gs.
        This gives a speedup of approximately (ecut_gs / ecut_pair)^{3/2}.

        Parameters
        ----------
        ecut_pair : float
            Plane-wave energy cutoff (in Hartree) for the pair density
            convolution. Must satisfy ecut_resp <= ecut_pair <= ecut_gs.
        """
        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2
        gs = self.gs
        N_c = np.array(qpd.gd.N_c)
        N_tot = int(np.prod(N_c))
        Omega = abs(np.linalg.det(qpd.gd.cell_cv))

        # --- Determine coarse grid dimensions ---
        # The coarse grid resolves wave functions up to ecut_pair.
        # Scale from the fine grid, which resolves up to ecut_gs.
        ratio = np.sqrt(ecut_pair / gs.pd.ecut)
        M_c = np.maximum(4, np.round(N_c * ratio).astype(int))
        M_c += M_c % 2  # ensure even for FFT efficiency
        M_tot = int(np.prod(M_c))
        dv_coarse = Omega / M_tot
        pw_scale = M_tot / N_tot  # rescale PW coefficients for coarse grid

        # --- IBZ k-point data ---
        kd = gs.kd
        ik1 = kd.bz2ibz_k[kpt1.K]
        ik2 = kd.bz2ibz_k[kpt2.K]
        ibz2bz1 = gs.ibz2bz[kpt1.K]
        ibz2bz2 = gs.ibz2bz[kpt2.K]
        psit1_all = gs.kpt_ks[ik1][kpt1.s].psit_nG
        psit2_all = gs.kpt_ks[ik2][kpt2.s].psit_nG

        # --- Truncation masks: keep only G with ½|G+k|² < ecut_pair ---
        G1_Gv = gs.pd.get_reciprocal_vectors(q=ik1, add_q=True)
        G2_Gv = gs.pd.get_reciprocal_vectors(q=ik2, add_q=True)
        mask1 = 0.5 * np.sum(G1_Gv**2, axis=1) < ecut_pair
        mask2 = 0.5 * np.sum(G2_Gv**2, axis=1) < ecut_pair

        # --- Map IBZ G-vectors to coarse grid (with symmetry rotation) ---
        def _ibz_to_coarse(Q_fine, mask, U_cc, time_reversal):
            i_cG = np.array(np.unravel_index(Q_fine[mask], N_c))
            if time_reversal:
                i_cG = -(U_cc @ i_cG)
            else:
                i_cG = U_cc @ i_cG
            return np.ravel_multi_index(
                _fine_to_coarse_3d(i_cG, N_c, M_c), M_c)

        Q1_coarse = _ibz_to_coarse(gs.pd.Q_qG[ik1], mask1,
                                    ibz2bz1.U_cc, ibz2bz1.time_reversal)
        Q2_coarse = _ibz_to_coarse(gs.pd.Q_qG[ik2], mask2,
                                    ibz2bz2.U_cc, ibz2bz2.time_reversal)

        # --- Map shifted response G-vectors to coarse grid ---
        shift_c = np.round(kpt1.k_c + qpd.q_c - kpt2.k_c).astype(int)
        i_resp = np.array(np.unravel_index(qpd.Q_qG[0], N_c))
        i_resp_shifted = i_resp + shift_c[:, np.newaxis]
        Q_resp_coarse = np.ravel_multi_index(
            _fine_to_coarse_3d(i_resp_shifted, N_c, M_c), M_c)

        # --- Helper: construct one coarse-grid wavefunction ---
        def _make_coarse_wf(psit_G, mask, Q_coarse, time_reversal):
            coeffs = psit_G[mask] * pw_scale
            if time_reversal:
                coeffs = coeffs.conj()
            buf = np.zeros(tuple(M_c), dtype=complex)
            buf.ravel()[Q_coarse] = coeffs
            return np.fft.ifftn(buf)

        # --- Pre-load coarse-grid K2 wavefunctions ---
        myblocksize = kpt2.nb - kpt2.na
        ut2_coarse = np.empty((myblocksize,) + tuple(M_c), dtype=complex)
        with self.context.timer('coarse wf K2'):
            for m_local in range(myblocksize):
                m_global = kpt2.na + m_local
                ut2_coarse[m_local] = _make_coarse_wf(
                    psit2_all[m_global], mask2,
                    Q2_coarse, ibz2bz2.time_reversal)

        # --- Compute pair densities ---
        nG = qpd.ngmax
        if output_buffer is None:
            n_nmG = np.zeros((len(n_n), len(m_m), nG), qpd.dtype)
        else:
            n_nmG = output_buffer

        for j, n in enumerate(n_n):
            with self.context.timer('coarse wf K1'):
                ut1cc_coarse = _make_coarse_wf(
                    psit1_all[n], mask1,
                    Q1_coarse, ibz2bz1.time_reversal).conj()

            with self.context.timer('paw'):
                C1_aGi = pawcorr.multiply(kpt1.P_ani, band=n - kpt1.na)

            n_mG = qpd.empty(kpt2.blocksize)
            n_mG[:] = 0.0

            with self.context.timer('coarse pair FFT'):
                for m_local in range(myblocksize):
                    product = ut1cc_coarse * ut2_coarse[m_local]
                    F = np.fft.fftn(product)
                    n_mG[m_local] = F.ravel()[Q_resp_coarse] * dv_coarse

            # PAW corrections (unchanged from fine-grid approach)
            with self.context.timer('gemm'):
                for C1_Gi, P2_mi in zip(C1_aGi, kpt2.P_ani):
                    mmm(1.0, P2_mi, 'N', C1_Gi, 'T', 1.0,
                        n_mG[:myblocksize])

            if not block or self.blockcomm.size == 1:
                n_nmG[j] = n_mG
            else:
                n_MG = qpd.empty(kpt2.blocksize * self.blockcomm.size)
                with self.context.timer('all_gather'):
                    self.blockcomm.all_gather(n_mG, n_MG)
                n_nmG[j] = n_MG[:kpt2.n2 - kpt2.n1]

        return n_nmG

    @timer('get_optical_pair_density_head')
    def get_optical_pair_density_head(self, qpd, kptpair, n_n, m_m,
                                      block=False, output_buffer=None):
        """Get the optical limit of the pair density head (G=0) for a k-pair.
        """
        assert np.allclose(qpd.q_c, 0.0), f"{qpd.q_c} is not the optical limit"

        kpt1 = kptpair.kpt1
        kpt2 = kptpair.kpt2

        if output_buffer is None:
            # v = (x, y, z)
            n_nmv = np.zeros((len(n_n), len(m_m), 3), qpd.dtype)
        else:
            assert output_buffer.shape == (len(n_n), len(m_m), 3)
            assert output_buffer.dtype == qpd.dtype
            n_nmv = output_buffer

        for j, n in enumerate(n_n):
            n_nmv[j] = self.calculate_optical_pair_density_head(n, m_m,
                                                                kpt1, kpt2,
                                                                block=block)

        return n_nmv

    @timer('Calculate pair-densities')
    def calculate_pair_density(self, ut1cc_R, C1_aGi, kpt2, qpd, Q_G,
                               block=True):
        """Calculate FFT of pair-densities and add PAW corrections.

        ut1cc_R: 3-d complex ndarray
            Complex conjugate of the periodic part of the left hand side
            wave function.
        C1_aGi: list of ndarrays
            PAW corrections for all atoms.
        kpt2: KPoint object
            Right hand side k-point object.
        qpd: SingleQPWDescriptor
            Plane-wave descriptor for q=k2-k1.
        Q_G: 1-d int ndarray
            Mapping from flattened 3-d FFT grid to 0.5(G+q)^2<ecut sphere.
        """
        dv = qpd.gd.dv
        n_mG = qpd.empty(kpt2.blocksize)
        myblocksize = kpt2.nb - kpt2.na

        for ut_R, n_G in zip(kpt2.ut_nR, n_mG):
            n_R = ut1cc_R * ut_R
            with self.context.timer('fft'):
                n_G[:] = qpd.fft(n_R, 0, Q_G) * dv
        # PAW corrections:
        with self.context.timer('gemm'):
            for C1_Gi, P2_mi in zip(C1_aGi, kpt2.P_ani):
                # gemm(1.0, C1_Gi, P2_mi, 1.0, n_mG[:myblocksize], 't')
                mmm(1.0, P2_mi, 'N', C1_Gi, 'T', 1.0, n_mG[:myblocksize])

        if not block or self.blockcomm.size == 1:
            return n_mG
        else:
            n_MG = qpd.empty(kpt2.blocksize * self.blockcomm.size)
            with self.context.timer('all_gather'):
                self.blockcomm.all_gather(n_mG, n_MG)
            return n_MG[:kpt2.n2 - kpt2.n1]

    @timer('Optical limit')
    def calculate_optical_pair_velocity(self, n, kpt1, kpt2, block=False):
        # This has the effect of caching at most one kpoint.
        # This caching will be efficient only if we are looping over kpoints
        # in a particular way.
        #
        # It would be better to refactor so this caching is handled explicitly
        # by the caller providing the right thing.
        #
        # See https://gitlab.com/gpaw/gpaw/-/issues/625
        if self.ut_sKnvR is None or kpt1.K not in self.ut_sKnvR[kpt1.s]:
            self.ut_sKnvR = self.calculate_derivatives(kpt1)

        gd = self.gs.gd
        k_v = 2 * np.pi * np.dot(kpt1.k_c, np.linalg.inv(gd.cell_cv).T)

        ut_vR = self.ut_sKnvR[kpt1.s][kpt1.K][n - kpt1.n1]
        atomdata_a = self.gs.pawdatasets.by_atom
        C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[n - kpt1.na])
                 for atomdata, P_ni in zip(atomdata_a, kpt1.P_ani)]

        blockbands = kpt2.nb - kpt2.na
        n0_mv = np.empty((kpt2.blocksize, 3), dtype=complex)
        nt_m = np.empty(kpt2.blocksize, dtype=complex)
        n0_mv[:blockbands] = -self.gs.gd.integrate(ut_vR,
                                                   kpt2.ut_nR).T
        nt_m[:blockbands] = self.gs.gd.integrate(kpt1.ut_nR[n - kpt1.na],
                                                 kpt2.ut_nR)

        n0_mv[:blockbands] += (1j * nt_m[:blockbands, np.newaxis] *
                               k_v[np.newaxis, :])

        for C_vi, P_mi in zip(C_avi, kpt2.P_ani):
            # gemm(1.0, C_vi, P_mi, 1.0, n0_mv[:blockbands], 'c')
            mmm(1.0, P_mi, 'N', C_vi, 'C', 1.0, n0_mv[:blockbands])

        if block and self.blockcomm.size > 1:
            n0_Mv = np.empty((kpt2.blocksize * self.blockcomm.size, 3),
                             dtype=complex)
            with self.context.timer('all_gather optical'):
                self.blockcomm.all_gather(n0_mv, n0_Mv)
            n0_mv = n0_Mv[:kpt2.n2 - kpt2.n1]

        return -1j * n0_mv

    def calculate_optical_pair_density_head(self, n, m_m, kpt1, kpt2,
                                            block=False):
        # Numerical threshold for the optical limit k dot p perturbation
        # theory expansion:
        threshold = 1

        eps1 = kpt1.eps_n[n - kpt1.n1]
        deps_m = (eps1 - kpt2.eps_n)[m_m - kpt2.n1]
        n0_mv = self.calculate_optical_pair_velocity(n, kpt1, kpt2,
                                                     block=block)

        deps_m = deps_m.copy()
        deps_m[deps_m == 0.0] = np.inf

        smallness_mv = np.abs(-1e-3 * n0_mv / deps_m[:, np.newaxis])
        inds_mv = (np.logical_and(np.inf > smallness_mv,
                                  smallness_mv > threshold))
        n0_mv *= - 1 / deps_m[:, np.newaxis]
        n0_mv[inds_mv] = 0

        return n0_mv

    @timer('Intraband')
    def intraband_pair_density(self, kpt, n_n):
        """Calculate intraband matrix elements of nabla"""
        # Bands and check for block parallelization
        na, nb, n1 = kpt.na, kpt.nb, kpt.n1
        vel_nv = np.zeros((nb - na, 3), dtype=complex)
        assert np.max(n_n) < nb, 'This is too many bands'
        assert np.min(n_n) >= na, 'This is too few bands'

        # Load kpoints
        gd = self.gs.gd
        k_v = 2 * np.pi * np.dot(kpt.k_c, np.linalg.inv(gd.cell_cv).T)
        atomdata_a = self.gs.pawdatasets.by_atom

        # Break bands into degenerate chunks
        degchunks_cn = []  # indexing c as chunk number
        for n in n_n:
            inds_n = np.nonzero(np.abs(kpt.eps_n[n - n1] -
                                       kpt.eps_n) < 1e-5)[0] + n1

            # Has this chunk already been computed?
            oldchunk = any([n in chunk for chunk in degchunks_cn])
            if not oldchunk:
                if not all([ind in n_n for ind in inds_n]):
                    raise RuntimeError(
                        'You are cutting over a degenerate band '
                        'using block parallelization.')
                degchunks_cn.append(inds_n)

        # Calculate matrix elements by diagonalizing each block
        for ind_n in degchunks_cn:
            deg = len(ind_n)
            ut_nvR = self.gs.gd.zeros((deg, 3), complex)
            vel_nnv = np.zeros((deg, deg, 3), dtype=complex)
            # States are included starting from kpt.na
            ut_nR = kpt.ut_nR[ind_n - na]

            # Get derivatives
            for ind, ut_vR in zip(ind_n, ut_nvR):
                ut_vR[:] = self.make_derivative(kpt.s, kpt.K,
                                                ind, ind + 1)[0]

            # Treat the whole degenerate chunk
            for n in range(deg):
                ut_vR = ut_nvR[n]
                C_avi = [np.dot(atomdata.nabla_iiv.T, P_ni[ind_n[n] - na])
                         for atomdata, P_ni in zip(atomdata_a, kpt.P_ani)]

                nabla0_nv = -self.gs.gd.integrate(ut_vR, ut_nR).T
                nt_n = self.gs.gd.integrate(ut_nR[n], ut_nR)
                nabla0_nv += 1j * nt_n[:, np.newaxis] * k_v[np.newaxis, :]

                for C_vi, P_ni in zip(C_avi, kpt.P_ani):
                    # gemm(1.0, C_vi, P_ni[ind_n - na], 1.0, nabla0_nv, 'c')
                    mmm(1.0, P_ni[ind_n - na], 'N', C_vi, 'C', 1.0, nabla0_nv)

                vel_nnv[n] = -1j * nabla0_nv

            for iv in range(3):
                vel, _ = np.linalg.eig(vel_nnv[..., iv])
                vel_nv[ind_n - na, iv] = vel  # Use eigenvalues

        return vel_nv[n_n - na]

    def calculate_derivatives(self, kpt):
        ut_sKnvR = [{}, {}]
        ut_nvR = self.make_derivative(kpt.s, kpt.K, kpt.n1, kpt.n2)
        ut_sKnvR[kpt.s][kpt.K] = ut_nvR

        return ut_sKnvR

    @timer('Derivatives')
    def make_derivative(self, s, K, n1, n2):
        gs = self.gs
        U_cc = gs.ibz2bz[K].U_cc
        A_cv = gs.gd.cell_cv
        M_vv = np.dot(np.dot(A_cv.T, U_cc.T), np.linalg.inv(A_cv).T)
        ik = gs.kd.bz2ibz_k[K]
        assert gs.kd.comm.size == 1
        kpt = gs.kpt_ks[ik][s]
        psit_nG = kpt.psit_nG
        iG_Gv = 1j * gs.pd.get_reciprocal_vectors(q=ik, add_q=False)
        ut_nvR = gs.gd.zeros((n2 - n1, 3), complex)
        for n in range(n1, n2):
            for v in range(3):
                ut_R = gs.ibz2bz[K].map_pseudo_wave(
                    gs.pd.ifft(iG_Gv[:, v] * psit_nG[n], ik))
                for v2 in range(3):
                    ut_nvR[n - n1, v2] += ut_R * M_vv[v, v2]

        return ut_nvR


def phase_shifted_fft_indices(k1_c, k2_c, qpd, coordinate_transformation=None):
    """Get phase shifted FFT indices for G-vectors inside the cutoff sphere.

    The output 1D FFT indices Q_G can be used to extract the plane-wave
    components G of the phase shifted Fourier transform

    n_kk'(G+q) = FFT_G[e^(-i[k+q-k']r) n_kk'(r)]

    where n_kk'(r) is some lattice periodic function and the wave vector
    difference k + q - k' is commensurate with the reciprocal lattice.
    """
    N_c = qpd.gd.N_c
    Q_G = qpd.Q_qG[0]
    q_c = qpd.q_c
    if coordinate_transformation:
        q_c = coordinate_transformation(q_c)

    shift_c = k1_c + q_c - k2_c
    assert np.allclose(shift_c.round(), shift_c)
    shift_c = shift_c.round().astype(int)

    if shift_c.any() or coordinate_transformation:
        # Get the 3D FFT grid indices (relative reciprocal space coordinates)
        # of the G-vectors inside the cutoff sphere
        i_cG = np.unravel_index(Q_G, N_c)
        if coordinate_transformation:
            i_cG = coordinate_transformation(i_cG)
        # Shift the 3D FFT grid indices to account for the Bloch-phase shift
        # e^(-i[k+q-k']r)
        i_cG += shift_c[:, np.newaxis]
        # Transform back the FFT grid to 1D FFT indices
        Q_G = np.ravel_multi_index(i_cG, N_c, 'wrap')

    return Q_G


def get_gs_and_context(calc, txt, world, timer):
    """Interface to initialize gs and context from old input arguments.
    Should be phased out in the future!"""
    from gpaw.new.ase_interface import ASECalculator as NewGPAW
    from gpaw.old.calculator import GPAW as OldGPAW

    context = ResponseContext(txt=txt, timer=timer, comm=world)

    if isinstance(calc, (OldGPAW, NewGPAW)):
        assert calc.wfs.world.size == 1
        gs = calc.gs_adapter()
    else:
        gs = ResponseGroundStateAdapter.from_gpw_file(gpw=calc)

    return gs, context
