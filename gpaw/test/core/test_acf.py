import sys

import numpy as np
import pytest

from gpaw.core import PWDesc, UGDesc
from gpaw.gpu import cupy as cp
from gpaw.mpi import world

a = 2.5
n = 20


@pytest.fixture
def grid():
    # comm = world.new_communicator([world.rank])
    return UGDesc(cell=[a, a, a], size=(n, n, n), comm=world, dtype=complex)


# Gussian:
alpha = 4.0
s = (0, 3.0, lambda r: np.exp(-alpha * r**2))
gauss_integral = np.pi / 2 / alpha**1.5


@pytest.mark.ci
@pytest.mark.parametrize('xp', [np])
def test_acf_fd(grid, xp):

    basis = grid.atom_centered_functions(
        [[s]],
        positions=[[0.5, 0.5, 0.5]], xp=xp)
    coefs = basis.layout.empty()
    if 0 in coefs:
        print(coefs[0])
        coefs[0] = [1.0]
    f1 = grid.zeros(xp=xp)
    basis.add_to(f1, coefs.to_xp(xp))

    if 0:
        from sympy import exp, integrate, oo, var
        a0, r = var('a, r')
        integrate(exp(-a0 * r**2) * r**2, (r, 0, oo))

    assert f1.integrate() == pytest.approx(gauss_integral)

    f2 = f1.gather(broadcast=True)
    x, y = f2.xy(n // 2, n // 2, ...)
    y0 = np.exp(-alpha * (x - a / 2)**2) / (4 * np.pi)**0.5
    assert abs(y - y0).max() == pytest.approx(0.0, abs=0.001)


@pytest.mark.ci
@pytest.mark.gpu
@pytest.mark.parametrize('xp', [np, cp])
def test_acf_pw(grid, xp):
    if world.size > 1 and xp is cp:
        pytest.skip()
    if xp is cp and '_gpaw' in sys.builtin_module_names:
        pytest.skip()

    pw = PWDesc(ecut=50, cell=grid.cell, dtype=complex, comm=world)

    basis = pw.atom_centered_functions(
        [[s]],
        positions=[[0.5, 0.5, 0.5]], xp=xp)

    coefs = basis.layout.empty()
    if 0 in coefs:
        print(coefs[0])
        coefs[0] = xp.asarray([1.0])

    f1 = pw.zeros(xp=xp)
    basis.add_to(f1, coefs)
    assert f1.integrate() == pytest.approx(gauss_integral)
    f2 = f1.gather(broadcast=True)
    r2 = f2.ifft(grid=grid.new(comm=None))
    x, y = r2.xy(10, 10, ...)
    y0 = np.exp(-alpha * (x - a / 2)**2) / (4 * np.pi)**0.5
    assert abs(y - y0).max() == pytest.approx(0.0, abs=0.002)
