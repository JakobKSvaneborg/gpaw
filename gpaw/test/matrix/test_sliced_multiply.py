import numpy as np
import time
import pytest
import warnings

from gpaw.mpi import world


def test_sliced_multiply(N=10, max_mem=2e2):
    from gpaw.core.matrix import Matrix
    N = int(N * np.sqrt(world.size))

    A_nn = Matrix(
        N,
        N,
        dtype=np.complex128,
        data=None,
        dist=(world, world.size, 1),
        xp=np,
    )
    B_nX = Matrix(
        N,
        100 * N,
        dtype=np.complex128,
        data=None,
        dist=(world, world.size, 1),
        xp=np,
    )
    A_nn.data[:] = 1
    B_nX.data[:] = 1

    buffer_nx = np.empty(int(max_mem) * 2, np.byte)
    if buffer_nx.shape == B_nX.shape:
        # Only time the multiply
        then = time.time()
        A_nn.multiply(B_nX, out=buffer_nx)
        now = time.time()
        warnings.warn('Not testing sliced multiply')
    else:
        # Only time the multiply
        then = time.time()
        A_nn.multiply(B_nX, out=B_nX, data_buffer=buffer_nx)
        now = time.time()

    assert B_nX.data == pytest.approx(N)
    return now - then


if __name__ == '__main__':
    max_mem_l = [2e5, 2e6, 2e7, 2e10]
    times = [test_sliced_multiply(N=500, max_mem=mem) for mem in max_mem_l]
    print(max_mem_l, times)
