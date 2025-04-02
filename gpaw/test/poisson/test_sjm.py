from math import pi

import numpy as np
import pytest

from gpaw.core import PWDesc, UGDesc
from gpaw.new.pw.poisson import PWPoissonSolver
from gpaw.new.sjm import SJMPoissonSolver


def f(a, z, z0, w):
    return np.exp(-((z - z0) / w)**2) / a**2 / np.pi**0.5 / w


if 0:  # Analytic result
    from sympy import integrate, exp, oo, var, Symbol
    z = var('z')
    w = Symbol('w', positive=True)
    m = integrate(exp(-(z / w)**2), (z, -oo, oo))
    print(m)


def test_sjm():
    a = 1.0
    L = 8.0
    grid = UGDesc.from_cell_and_grid_spacing(cell=[a, a, L], grid_spacing=0.15)
    z = grid.xyz()[0, 0, :, 2]
    c = 0.05
    rhot = f(a, z, 4.0, 2.0) * (1 + c)
    rhot -= f(a, z, 4.0, 1.0)
    rhot -= f(a, z, 6.0, 1.0) * c
    eps = 1.0 + f(a, z, 5.0, 2.0) * 20
    rhot_r = grid.zeros()
    rhot_r.data[:] = rhot
    eps_r =  grid.zeros()
    eps_r.data[:] = eps
    print(rhot_r.integrate())

    psolver = WeightedFDPoissonSolver()
    psolver.set_dielectric(self.dielectric)
    psolver.set_grid_descriptor(self.grid._gd)
    PoissonSolverWrapper(psolver)


if __name__ == '__main__':
    test_sjm()
