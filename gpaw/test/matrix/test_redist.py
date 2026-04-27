import pytest
from gpaw.core.matrix import Matrix
from gpaw.mpi import world
import numpy as np


@pytest.mark.skipif(world.size == 1, reason='Only run in parallel')
def test_redist():
    D = 2
    r = world.rank // 2 * 2
    domain_comm = world.new_communicator(range(r, r + 2))
    B = world.size // 2
    r = world.rank % 2
    band_comm = world.new_communicator(range(r, world.size, 2))
    n = 8
    M = Matrix(n, n, dist=band_comm)
    M.data[:] = np.arange(*M.dist.my_row_range())[:, np.newaxis]
    M2 = Matrix(n, n,
                dist=(world, B, D, n // B, n),
                data=M.data if domain_comm.rank == 0 else None)
    M3 = Matrix(n, n, dist=(world, B, D, 2, 2))
    M2.redist(M3)
    M4 = Matrix(n, n, dist=(world, 1, 1))
    M3.redist(M4)
    print(M4.data)
    if world.rank == 0:
        assert (M4.data.T == np.arange(n)).all()


if __name__ == '__main__':
    test_redist()
