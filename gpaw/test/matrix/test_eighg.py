from time import time

import numpy as np
import pytest
import scipy.linalg as linalg
from gpaw.core.matrix import Matrix, suggest_blocking
from gpaw.mpi import broadcast_exception, world, serial_comm


@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('algo', ['l', 's'])
def test_eighl(dtype, algo):
    n = 5
    S0 = Matrix(n, n, dist=(world, 1, 1), dtype=dtype)
    S = S0.new(dist=(world, world.size, 1))
    H0 = S0.new()
    H = S.new()
    if world.rank == 0:
        S0.data[:] = np.eye(n)
        H0.data[:] = 0.0
        H0.data.ravel()[::n + 1] = np.arange(n) + 1
        if dtype == float:
            S0.data[-1, 0] = 0.001
            H0.data[-1, 0] = 0.001
        else:
            S0.data[-1, 0] = 0.001j
            H0.data[-1, 0] = 0.001j
    H0.tril2full()
    S0.tril2full()
    H00 = H0.copy()
    S0.redist(S)
    H0.redist(H)
    if world.rank == 0:
        eigs0, C0 = linalg.eigh(H0.data, S0.data)
        print(eigs0)
        print(C0)
        error = H0.data @ C0 - S0.data @ C0 @ np.diag(eigs0)
        print(error)

    if algo == 'l':
        L = S.copy()
        L.invcholesky()
        L0 = S0.new()
        L.redist(L0)
        eigs = H.eighl(L)
        H.redist(H0)
    else:
        eigs = H.eigh(S)
        H.redist(H0)

    print(eigs)
    print(H0.data)
    with broadcast_exception(world):
        if world.rank == 0:
            if algo == 's':
                H0.data[:] = H0.data.T
            assert abs(eigs - eigs0).max() < 1e-14
            error = H00.data @ H0.data - S0.data @ H0.data @ np.diag(eigs)
            assert abs(error).max() < 1e-14


def benchmark(n, comm):
    S = Matrix(n, n, dist=(comm, comm.size, 1))
    H = S.new()
    S.data[:] = 0.0
    H.data[:] = 0.0
    n1, n2 = H.dist.my_row_range()
    S.data.flat[n1::n + 1] = 1.0
    H.data.flat[n1::n + 1] = np.linspace(1.0 + n1 / n,
                                         1.0 + (n2 - 1) / n,
                                         n2 - n1)
    if n1 == 0:
        S.data.flat[n::n + 1] = 0.1
        H.data.flat[n::n + 1] = 0.1
    else:
        S.data.flat[n1 - 1::n + 1] = 0.1
        H.data.flat[n1 - 1::n + 1] = 0.1

    sl = (comm, *suggest_blocking(n, comm.size))
    t1 = time()
    H.eigh(scalapack=sl)
    t1 = time() - t1
    t2 = time()
    H.eigh(S, scalapack=sl)
    t2 = time() - t2
    return t1, t2


if __name__ == '__main__':
    if world.rank == 0:
        fd = open(f'eigh-{world.size}.csv', 'w')
    for n in range(500, 2500, 500):
        t1, t2 = benchmark(n, comm=world)
        if world.rank == 0:
            t3, t4 = benchmark(n, comm=serial_comm)
            print(f'{n}, {t1}, {t2}, {t3 / t1}, {t4 / t2}', file=fd)
            fd.flush()
