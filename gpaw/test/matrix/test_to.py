import pytest
from gpaw.core.matrix import Matrix
from gpaw.mpi import world
import numpy as np
from gpaw.gpu import cupy as cp


@pytest.mark.parametrize('xp', [np, cp])
def test_to_xp_dtype(xp):
    N = 21
    m1 = Matrix(N, N, dist=world)
    m1.data[:] = world.rank
    xp2 = np if xp is cp else cp
    m2 = m1.to_xp(xp2).to_xp(xp)
    assert (m2.data == m1.data).all()
    m3 = m1.to_dtype(np.float32).to_dtype(float)
    assert (m3.data == m1.data).all()
