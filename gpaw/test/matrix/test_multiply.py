import pytest
import numpy as np

import gpaw.utilities.blas as blas
from gpaw.core.matrix import Matrix
from gpaw.core.matrix import mmm_nc, mmm_nc_sym, mmm_nn
from gpaw.mpi import world


def test_mmm():
    N = 3
    M = 5
    A_nX = Matrix(
        N,
        M,
        dtype=np.complex128,
        data=None,
        dist=(world, world.size, 1),
        xp=np,
    )
    A_nX.data[:] = 1
    B_nX = A_nX.copy()

    C_nn = A_nX.multiply(B_nX, opb='C', symmetric=True)
    C_nn.tril2full()
    assert C_nn.data == pytest.approx(M)
    C_nn.data[:] = 0
    mmm_nc_sym(A_nX, B_nX, C_nn, 1, blas.mmm)
    C_nn.tril2full()
    assert C_nn.data == pytest.approx(M)

    assert A_nX.multiply(B_nX, opb='C', symmetric=False).data \
        == pytest.approx(M)
    assert mmm_nc(A_nX, B_nX, C_nn, 1, 0, blas.mmm).data == pytest.approx(M)

    C_Xn = Matrix(
        M,
        N,
        dtype=np.complex128,
        data=None,
        dist=(world, world.size, 1),
        xp=np,
    )
    C_Xn.data[:] = 1

    assert A_nX.multiply(C_Xn).data == pytest.approx(M)
    mmm_nn(A_nX, C_Xn, C_nn, 1.0, 0.0, blas.mmm)
    assert C_nn.data == pytest.approx(M)

    D_nX = C_nn.multiply(C_Xn, opb='C')
    D_nX.multiply(C_Xn, beta=1.0, out=C_nn)

    assert C_nn.data == pytest.approx(M * M * N + M)
