from functools import cached_property

import numpy as np

from gpaw.response.gamma_int import GammaIntegral
from gpaw.response.pw_parallelization import Blocks1D


class DielectricFunctionCalculator:
    def __init__(self, chi0, coulomb, xckernel, mode):
        self.coulomb = coulomb
        self.qpd = chi0.qpd
        self.mode = mode
        self.optical_limit = chi0.optical_limit
        self.chi0 = chi0
        self.xckernel = xckernel
        self.wblocks = Blocks1D(chi0.body.blockdist.blockcomm, len(chi0.wd))
        # Generate fine grid in vicinity of gamma
        if chi0.optical_limit and self.wblocks.nlocal:
            self.gamma_int = GammaIntegral(self.coulomb, self.qpd)
        else:
            self.gamma_int = None

    @cached_property
    def sqrtV_G(self):
        return self.coulomb.sqrtV(qpd=self.qpd, q_v=None)

    @cached_property
    def I_GG(self):
        return np.eye(self.qpd.ngmax)

    @cached_property
    def fxc_GG(self):
        if self.mode == 'GW':
            return self.I_GG
        else:
            return self.xckernel.calculate(self.qpd)

    def get_epsinv_wGG(self, only_correlation=True):
        """
        Calculates inverse dielectric matrix for all frequencies.
        """
        epsinv_wGG = np.zeros((self.wblocks.nlocal, *self.I_GG.shape),
                              dtype=complex)

        # chi0_wGG cannot be a cached property since some cores do not have
        # any frequency points. Therefore we get it here.
        chi0_wGG = self.chi0.body.array_with_distribution('wGG')

        for w, epsinv_GG in enumerate(epsinv_wGG):
            epsinv_GG[:] = self.single_frequency_epsinv_GG(w, chi0_wGG)
            if only_correlation:
                epsinv_GG -= self.I_GG
        return epsinv_wGG

    def single_frequency_epsinv_GG(self, w, chi0_wGG):
        """
        Calculates inverse dielectric matrix for single frequency
        """
        _dfc = _DielectricFunctionCalculator(self.sqrtV_G,
                                             chi0_wGG[w],
                                             self.mode,
                                             self.fxc_GG)
        if self.optical_limit:
            W = self.wblocks.a + w
            _dfc = _GammaDielectricFunctionCalculator(
                _dfc, self.gamma_int,
                self.chi0.chi0_Wvv[W], self.chi0.chi0_WxvG[W])
        return _dfc.get_epsinv_GG()


class _DielectricFunctionCalculator:
    def __init__(self, sqrtV_G, chi0_GG, mode, fxc_GG=None):
        self.sqrtV_G = sqrtV_G
        self.chiVV_GG = chi0_GG * sqrtV_G * sqrtV_G[:, np.newaxis]

        self.I_GG = np.eye(len(sqrtV_G))

        self.fxc_GG = fxc_GG
        self.chi0_GG = chi0_GG
        self.mode = mode

    def new_with(self, *, sqrtV_G, chi0_GG):
        return _DielectricFunctionCalculator(
            sqrtV_G, chi0_GG, self.mode, fxc_GG=self.fxc_GG)

    def _chiVVfxc_GG(self):
        assert self.mode != 'GW'
        assert self.fxc_GG is not None
        return self.chiVV_GG @ self.fxc_GG

    def eps_GG_gwp(self):
        gwp_inv_GG = np.linalg.inv(self.I_GG - self._chiVVfxc_GG() +
                                   self.chiVV_GG)
        return self.I_GG - gwp_inv_GG @ self.chiVV_GG

    def eps_GG_gws(self):
        # Note how the signs are different wrt. gwp.
        # Nobody knows why.
        gws_inv_GG = np.linalg.inv(self.I_GG + self._chiVVfxc_GG() -
                                   self.chiVV_GG)
        return gws_inv_GG @ (self.I_GG - self.chiVV_GG)

    def eps_GG_plain(self):
        return self.I_GG - self.chiVV_GG

    def eps_GG_w_fxc(self):
        return self.I_GG - self._chiVVfxc_GG()

    def get_eps_GG(self):
        mode = self.mode
        if mode == 'GWP':
            return self.eps_GG_gwp()
        elif mode == 'GWS':
            return self.eps_GG_gws()
        elif mode == 'GW':
            return self.eps_GG_plain()
        elif mode == 'GWG':
            return self.eps_GG_w_fxc()
        raise ValueError(f'Unknown mode: {mode}')

    def get_epsinv_GG(self):
        eps_GG = self.get_eps_GG()
        return np.linalg.inv(eps_GG)


class _GammaDielectricFunctionCalculator:

    def __init__(self, _dfc, gamma_int, chi0_vv, chi0_xvG):
        self._dfc = _dfc
        self.gamma_int = gamma_int

        self.chi0_vv = chi0_vv
        self.chi0_xvG = chi0_xvG

    @property
    def chi0_GG(self):
        return self._dfc.chi0_GG

    def get_epsinv_GG(self):
        # Get average epsinv over small region around Gamma
        if self._dfc.mode == 'GW':
            return self._get_epsinv_GG_woodbury()
        epsinv_GG = np.zeros(self.chi0_GG.shape, complex)
        for qweight, sqrtV_G, chi0_mapping in self.gamma_int:
            chi0p_GG = chi0_mapping(self.chi0_GG, self.chi0_vv, self.chi0_xvG)
            _dfc = self._dfc.new_with(sqrtV_G=sqrtV_G, chi0_GG=chi0p_GG)
            epsinv_GG += qweight * _dfc.get_epsinv_GG()
        return epsinv_GG

    def _get_epsinv_GG_woodbury(self):
        """Average epsinv over the Gamma region using rank-2 updates.

        For each optical wave vector q_f in the Gamma-point integration
        domain, eps(q_f) differs from the base dielectric matrix only in
        row 0 and column 0 (the Coulomb kernel of the remaining G-vectors
        is unaffected by the tiny |q_f| ~ 1e-6 offset).  Thus, instead of
        performing a full O(nG^3) inversion per integration point, the
        base matrix is inverted once, and each eps(q_f)^-1 is obtained
        through the Sherman-Morrison-Woodbury identity at O(nG^2) cost.
        """
        gamma_int = self.gamma_int
        A_GG = self._dfc.eps_GG_plain()
        Ainv_GG = np.linalg.inv(A_GG)

        epsinv_GG = Ainv_GG.copy()
        for q in range(len(gamma_int)):
            qweight, qf_v = gamma_int.integral_domain[q]
            sqrtV_G = gamma_int.coulomb.sqrtV(qpd=gamma_int.qpd, q_v=qf_v)

            # Row 0 and column 0 of chi0(q_f)
            head = qf_v @ self.chi0_vv @ qf_v
            row_G = qf_v @ self.chi0_xvG[0]
            col_G = qf_v @ self.chi0_xvG[1]
            row_G[0] = head
            col_G[0] = head

            # eps(q_f) = A + e_0 a^T + b e_0^T with b[0] = 0
            a_G = -sqrtV_G[0] * row_G * sqrtV_G - A_GG[0]
            a_G[0] += 1.0
            b_G = -sqrtV_G * col_G * sqrtV_G[0] - A_GG[:, 0]
            b_G[0] = 0.0

            # Woodbury update with U = [e_0, b], V = [a, e_0]
            AinvU_G2 = np.empty((len(a_G), 2), complex)
            AinvU_G2[:, 0] = Ainv_GG[:, 0]
            AinvU_G2[:, 1] = Ainv_GG @ b_G
            VtAinv_2G = np.empty((2, len(a_G)), complex)
            VtAinv_2G[0] = a_G @ Ainv_GG
            VtAinv_2G[1] = Ainv_GG[0]
            M_22 = np.eye(2, dtype=complex)
            M_22[0, 0] += a_G @ Ainv_GG[:, 0]
            M_22[0, 1] += a_G @ AinvU_G2[:, 1]
            M_22[1, 0] += Ainv_GG[0, 0]
            M_22[1, 1] += Ainv_GG[0] @ b_G
            epsinv_GG -= qweight * (
                AinvU_G2 @ np.linalg.solve(M_22, VtAinv_2G))
        return epsinv_GG
