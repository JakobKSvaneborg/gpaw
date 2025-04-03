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
    L = 10.0
    grid = UGDesc.from_cell_and_grid_spacing(cell=[a, a, L], grid_spacing=0.15)
    z = grid.xyz()[0, 0, :, 2]
    c = 0.05
    rhot = f(a, z, 4.0, 1.0) * (1 + c)
    rhot -= f(a, z, 4.0, 0.5)
    rhot -= f(a, z, 6.0, 1.0) * c
    eps = 1.0 + f(a, z, 5.0, 2.0) * 20
    rhot_r = grid.zeros()
    rhot_r.data[:] = rhot
    eps_r = grid.zeros()
    eps_r.data[:] = eps
    print(rhot_r.integrate())
    import matplotlib.pyplot as plt
    plt.plot(z, rhot_r.data[0, 0])
    pw = PWDesc(ecut=grid.ekin_max(), cell=grid.cell)
    ps = PWPoissonSolver(pw)
    vt_g = pw.zeros()
    rhot_g = rhot_r.fft(pw=pw)
    ps.solve(vt_g, rhot_g)
    vt_r = vt_g.ifft(grid=grid)
    plt.plot(z, vt_r.data[0, 0])
    plt.show()


if __name__ == '__main__':
    test_sjm()
