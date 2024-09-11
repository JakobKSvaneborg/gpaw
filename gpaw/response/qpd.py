from math import pi
import numpy as np

from gpaw.kpt_descriptor import KPointDescriptor
from gpaw.pw.descriptor import PWDescriptor
import gpaw.fftw as fftw


class SingleQPWDescriptor(PWDescriptor):
    @staticmethod
    def from_q(q_c, ecut, gd, gammacentered=False):
        """Construct a plane wave descriptor for q_c with a given cutoff."""
        qd = KPointDescriptor([q_c])
        if not isinstance(ecut, dict):
            return SingleQPWDescriptor(ecut, gd, complex, qd,
                                       gammacentered=gammacentered)
        else:
            return ecut['class'](gd=gd, kd=qd, gammacentered=gammacentered,
                                 dtype=complex, **ecut['kwargs'])

    @property
    def q_c(self):
        return self.kd.bzk_kc[0]

    @property
    def optical_limit(self):
        return np.allclose(self.q_c, 0.0)

    def copy(self):
        return self.copy_with()

    def copy_with(self, ecut=None, gd=None, gammacentered=None):
        if ecut is None:
            ecut = self.ecut
        if gd is None:
            gd = self.gd
        if gammacentered is None:
            gammacentered = self.gammacentered

        return SingleQPWDescriptor.from_q(
            self.q_c, ecut, gd, gammacentered=gammacentered)


class SingleCylQPWDescriptor(SingleQPWDescriptor):
    def __init__(self, ecut_xy, ecut_z, gd, dtype=None, kd=None,
                 fftwflags=fftw.MEASURE, gammacentered=False):

        assert gd.pbc_c.all()

        self.gd = gd
        self.fftwflags = fftwflags

        N_c = gd.N_c
        self.comm = gd.comm

        ecut0 = 0.5 * pi**2 / (self.gd.h_cv**2).sum(1).max()
        if ecut_xy is None:
            ecut_xy = 0.9999 * ecut0
        else:
            assert ecut_xy <= ecut0

        if ecut_z is None:
            ecut_z = 0.9999 * ecut0
        else:
            assert ecut_z <= ecut0

        self.ecut = ecut_xy
        self.ecut_xy = ecut_xy
        self.ecut_z = ecut_z

        if dtype is None:
            if kd is None or kd.gamma:
                dtype = float
            else:
                dtype = complex
        self.dtype = dtype
        self.gammacentered = gammacentered

        if dtype == float:
            Nr_c = N_c.copy()
            Nr_c[2] = N_c[2] // 2 + 1
            i_Qc = np.indices(Nr_c).transpose((1, 2, 3, 0))
            i_Qc[..., :2] += N_c[:2] // 2
            i_Qc[..., :2] %= N_c[:2]
            i_Qc[..., :2] -= N_c[:2] // 2
            self.tmp_Q = fftw.empty(Nr_c, complex)
            self.tmp_R = self.tmp_Q.view(float)[:, :, :N_c[2]]
        else:
            i_Qc = np.indices(N_c).transpose((1, 2, 3, 0))
            i_Qc += N_c // 2
            i_Qc %= N_c
            i_Qc -= N_c // 2
            self.tmp_Q = fftw.empty(N_c, complex)
            self.tmp_R = self.tmp_Q

        self.fftplan = fftw.create_plan(self.tmp_R, self.tmp_Q, -1, fftwflags)
        self.ifftplan = fftw.create_plan(self.tmp_Q, self.tmp_R, 1, fftwflags)

        # Calculate reciprocal lattice vectors:
        B_cv = 2.0 * pi * gd.icell_cv
        i_Qc.shape = (-1, 3)
        self.G_Qv = np.dot(i_Qc, B_cv)

        self.kd = kd
        if kd is None:
            self.K_qv = np.zeros((1, 3))
            self.only_one_k_point = True
        else:
            self.K_qv = np.dot(kd.ibzk_qc, B_cv)
            self.only_one_k_point = (kd.nbzkpts == 1)

        # Map from vectors inside sphere to fft grid:
        self.Q_qG = []
        G2_qG = []
        Q_Q = np.arange(len(i_Qc), dtype=np.int32)

        self.ng_q = []
        for q, K_v in enumerate(self.K_qv):
            G2_Q = ((self.G_Qv + K_v)**2).sum(axis=1)
            if gammacentered:
                mask_Q = ((self.G_Qv[:, 0:2]**2).sum(axis=1) <= 2 * ecut_xy) \
                    & ((self.G_Qv[:, 2]**2) <= 2 * ecut_z)
            else:
                G3_Q = ((self.G_Qv[:, 0:2] + K_v[0:2])**2).sum(axis=1)
                mask_Q = (G3_Q <= (2 * ecut_xy)) \
                    & ((self.G_Qv[:, 2]**2) <= (2 * ecut_z))

            if self.dtype == float:
                mask_Q &= ((i_Qc[:, 2] > 0) |
                           (i_Qc[:, 1] > 0) |
                           ((i_Qc[:, 0] >= 0) & (i_Qc[:, 1] == 0)))
            Q_G = Q_Q[mask_Q]
            self.Q_qG.append(Q_G)
            G2_qG.append(G2_Q[Q_G])
            ng = len(Q_G)
            self.ng_q.append(ng)

        self.ngmin = min(self.ng_q)
        self.ngmax = max(self.ng_q)

        if kd is not None:
            self.ngmin = kd.comm.min_scalar(self.ngmin)
            self.ngmax = kd.comm.max_scalar(self.ngmax)

        # Distribute things:
        S = gd.comm.size
        self.maxmyng = (self.ngmax + S - 1) // S
        ng1 = gd.comm.rank * self.maxmyng
        ng2 = ng1 + self.maxmyng

        self.G2_qG = []
        self.myQ_qG = []
        self.myng_q = []
        for q, G2_G in enumerate(G2_qG):
            G2_G = G2_G[ng1:ng2].copy()
            G2_G.flags.writeable = False
            self.G2_qG.append(G2_G)
            myQ_G = self.Q_qG[q][ng1:ng2]
            self.myQ_qG.append(myQ_G)
            self.myng_q.append(len(myQ_G))

        if S > 1:
            self.tmp_G = np.empty(self.maxmyng * S, complex)
        else:
            self.tmp_G = None
