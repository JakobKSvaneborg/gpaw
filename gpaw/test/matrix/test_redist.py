from gpaw.core.matrix import Matrix
from gpaw.mpi import world


def test_redist():
    D = 2
    r = world.rank // 2 * 2
    domain_comm = world.new_communicator(range(r, r + 2))
    B = world.size // 2
    r = world.rank % 2
    band_comm = world.new_communicator(range(r, world.size, 2))
    n = 8
    M = Matrix(n, n, dist=band_comm)
    M.data[:] = world.rank
    print('M', world.rank, band_comm.rank, domain_comm.rank, M)
    M2 = Matrix(n, n,
                dist=(world, B, D, n // B, n),
                data=M.data if domain_comm.rank == 0 else None)
    print(world.rank, band_comm.rank, domain_comm.rank, M2.data.flat[:])
    M3 = Matrix(n, n, dist=(world, B, D, 2, 2))
    print('M3', world.rank, band_comm.rank, domain_comm.rank, M3)
    M2.redist(M3)
    print(M3.data)


def test_redist0():
    n = 5
    M1 = Matrix(n, n,
                dist=(world, 2, 2, 2, 2))
    M1.data[:] = world.rank
    M2 = Matrix(n, n,
                dist=(world, 2, 2, n, n))
    M1.redist(M2)
    print(world.rank, M2.data.shape, M2.data)


if __name__ == '__main__':
    test_redist()
