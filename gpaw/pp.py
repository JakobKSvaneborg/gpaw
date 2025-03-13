"""Pure Python implementation of the _gpaw C-extension module.

Used if GPAW_NO_C_EXTENSION=1.  See also the gpaw.cgpaw module.
"""
import numpy as np
from scipy.interpolate import CubicSpline
from _gpaw import Operator

have_openmp = False

def gpaw_gpu_init():
    pass


def get_num_threads():
    return 1


class Spline:
    def __init__(self, l, rmax, f_g):
        self.spline = CubicSpline(np.linspace(0, rmax, len(f_g)), f_g,
                                  bc_type='clamped')
        self.l = l
        self.rmax = rmax

    def __call__(self, r):
        return self.spline(r)

    def get_angular_momentum_number(self):
        return self.l

    def get_cutoff(self):
        return self.rmax

    def map(self, r_g, out_g):
        out_g[:] = self.spline(r_g)
        out_g[r_g >= self.rmax] = 0.0


def hartree(l: int,
            nrdr: np.ndarray,
            r: np.ndarray,
            vr: np.ndarray) -> None:
    p = 0.0
    q = 0.0
    for g in range(len(r) - 1, 0, -1):
        R = r[g]
        rl = R**l
        dp = nrdr[g] / rl
        rlp1 = rl * R
        dq = nrdr[g] * rlp1
        vr[g] = (p + 0.5 * dp) * rlp1 - (q + 0.5 * dq) / rl
        p += dp
        q += dq
    vr[0] = 0.0
    f = 4.0 * np.pi / (2 * l + 1)
    vr[1:] += q / r[1:]**l
    vr[1:] *= f


def unpack(M, M2):
    n = len(M2)
    p = 0
    for r in range(n):
        for c in range(r, n):
            d = M[p]
            M2[r, c] = d
            M2[c, r] = d
            p += 1


def pack(M2):
    n = len(M2)
    M = np.empty(n * (n + 1) // 2)
    p = 0
    for r in range(n):
        M[p] = M2[r, r]
        p += 1
        for c in range(r + 1, n):
            M[p] = M2[r, c] + M2[c, r]
            p += 1
    return M
