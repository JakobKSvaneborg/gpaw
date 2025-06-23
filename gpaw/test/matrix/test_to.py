import pytest
from gpaw.core.matrix import Matrix
from gpaw.mpi import broadcast_exception, world
import numpy as np
import gpaw.gpu.cupy as cp


@pytest.mark.parametrize('xp', [np, cp]):
def test_to_xp_dtype(xp):
    N = 21
    m1 = Matrix(N, N, dist=world)
    xp2 = np if xp is cp else cp
    m2 = m1.to_xp(xp2).to_xp(xp2)
    m3 = m1.to_dtype(np.float32).to_dtype(float)
    assert ...
