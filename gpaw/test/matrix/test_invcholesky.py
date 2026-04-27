import numpy as np
import pytest

from gpaw.core.matrix import Matrix
from gpaw.mpi import broadcast_exception, world


@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.skipif(world.size > 2, reason='size>2')
def test_invcholesky(dtype):
    S0 = Matrix(2, 2, dist=(world, 1, 1), dtype=dtype)
    S = S0.new(dist=(world, world.size, 1))
    if world.rank == 0:
        if dtype == float:
            S0.data[:] = np.array([[1.0, 117], [0.1, 2.0]])
        else:
            S0.data[:] = np.array([[1.0, 117], [0.1j, 2.0]])
    S0.redist(S)
    L = S.copy()
    L.invcholesky()
    S.tril2full()
    if world.size == 1:
        print(L.data @ S.data @ L.data.T.conj())
    else:
        print(world.rank, L.data)
    A = L.multiply(S, opa='N').multiply(L, opb='C')
    print(A.data)
    A.redist(S0)
    with broadcast_exception(world):
        if world.rank == 0:
            assert abs(S0.data - np.eye(2)).max() < 1e-14
