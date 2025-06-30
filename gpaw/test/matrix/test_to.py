import pytest
from gpaw.core.matrix import Matrix
from gpaw.mpi import world
import numpy as np
from gpaw.gpu import cupy as cp


@pytest.mark.parametrize('xp', [np, cp])
def test_to_xp_dtype(xp):
    N = 21
    m1 = Matrix(N, N, dist=(world, -1, 1))
    m1.data[:] = world.rank
    m2 = m1.to_xp(xp)
    m3 = m2.to_xp(np)
    assert (m3.data == m1.data).all()
    m4 = m2.to_dtype(np.float32).to_dtype(float)
    assert (m4.data == m2.data).all()
