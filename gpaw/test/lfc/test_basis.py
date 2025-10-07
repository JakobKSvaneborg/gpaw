from typing import Callable

import numpy as np
import pytest
from gpaw.core import UGArray, UGDesc
from gpaw.core.matrix import Matrix
from gpaw.lfc import BasisFunctions
from gpaw.sphere.spherical_harmonics import Y
from gpaw.spline import Spline


def simple(vt_R: UGArray,
           basis_functions_a: list[list[tuple[int, float, Callable]]],
           pos_av: np.ndarray
           ) -> Matrix:
    """Simple, but slow, implementation of potential matrix-elements."""
    N = sum(2 * l + 1
            for basis_functions in basis_functions_a
            for l, _, _ in basis_functions)
    grid = vt_R.desc
    R_Rv = grid.xyz()
    M = 0
    phit_MR = grid.zeros(N)
    for R_v, basis_functions in zip(pos_av, basis_functions_a):
        d_Rv = R_Rv - R_v
        d_R = (d_Rv**2).sum(axis=3)**0.5
        for l, rc, func in basis_functions:
            mask_R = d_R < rc
            d_r = d_R[mask_R]
            d_rv = d_Rv[mask_R]
            f_r = func(d_r)
            for m in range(2 * l + 1):
                phit_MR.data[M + m, mask_R] = Y(l**2 + m, *d_rv.T) * f_r
            M += 2 * l + 1
    vtpsit_MR = phit_MR.copy()
    vtpsit_MR.data *= vt_R.data
    return Matrix(N, N, data=phit_MR.integrate(vtpsit_MR))


def ccode(vt_R: UGArray,
          basis_functions_a: list[list[tuple[int, float, Callable]]],
          pos_av: np.ndarray
          ) -> Matrix:
    grid = vt_R.desc
    basis = BasisFunctions(
        grid._gd,
        [[Spline.from_data(l, rc, func(np.linspace(0, rc, 100)))
          for l, rc, func in basis_functions]
         for basis_functions in basis_functions_a])
    basis.set_positions(np.linalg.solve(grid.cell_cv.T, pos_av.T).T)
    N = basis.Mmax
    V_MM = Matrix(N, N)
    basis.calculate_potential_matrix(vt_R.data, V_MM.data, 0)
    return V_MM


def func(r):
    """radial function (goes to 0.0 at r=1.0)."""
    return 1 - 3 * r**2 + 2 * r**3


@pytest.mark.serial
def test_pot_mat_elements():
    """Calculate overlap between s and p function."""
    n = 24
    a = 3.0
    grid = UGDesc(cell=[a, a, a], size=(n, n, n))
    vt_R = grid.zeros()
    vt_R.data += 0.5
    rc = 1.0
    basis_functions_a = [[(0, rc, func)], [(1, rc, func)]]
    pos_av = np.array([[1, 1, 1], [2.0, 2, 1]])
    V1_MM = simple(vt_R, basis_functions_a, pos_av)
    V2_MM = ccode(vt_R, basis_functions_a, pos_av)
    V2_MM.tril2full()
    assert V1_MM.data == pytest.approx(V2_MM.data)
