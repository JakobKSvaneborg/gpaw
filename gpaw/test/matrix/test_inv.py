import numpy as np
import pytest
from gpaw.core.matrix import Matrix
from gpaw.mpi import broadcast_exception, world


@pytest.mark.parametrize('dtype', [float, complex])
def test_inv(dtype):
    if world.size > 1 and dtype == float:
        pytest.skip('Not implemented')

    N = 15
    S0 = Matrix(N, N,
                dist=(world, 1, 1),
                dtype=dtype)

    if world.rank == 0:
        S0.data[:] = np.diag(np.arange(1, N + 1))
        if dtype == float:
            S0.data[-1, 0] = 0.1
        else:
            S0.data[-1, 0] = 0.1j

    S = S0.new(dist=(world, world.size, 1, 2))
    S0.redist(S)

    iS = S.copy()
    iS.inv()

    S.tril2full()
    iS.tril2full()

    A = S.multiply(iS)
    A.redist(S0)

    with broadcast_exception(world):
        if world.rank == 0:
            assert abs(S0.data - np.eye(N)).max() < 1e-14


if __name__ == '__main__':
    test_inv(complex)
