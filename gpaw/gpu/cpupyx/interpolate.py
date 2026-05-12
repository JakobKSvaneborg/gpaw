import gpaw.gpu.cpupy as cp
from scipy.interpolate import PPoly as ScipyPPoly


class PPoly:
    """"""

    def __init__(self, c: cp.ndarray, x: cp.ndarray, extrapolate=None,
                 axis=0):
        """"""
        self.impl = ScipyPPoly(c._data, x._data, extrapolate=extrapolate,
                               axis=axis)

    def __call__(self, x: cp.ndarray, nu=0, extrapolate=None) -> cp.ndarray:
        """"""
        return cp.asarray(self.impl(x._data))
