from gpaw.core.matrix import Matrix
from gpaw.mpi import world
import numpy as np


def test_redist():
    m1 = Matrix(2, 2, dist=(world, 1, 4, 2))
    if world.rank == 0:
        m1.data.flat[:] = [2, 0, 0, 3]
    comm = world.new_communicator([0, 1])
    m1.eigh(scalapack=(comm, 2, 1, 1))
    # m1.redist(m2)
    # print(m2.data)


if __name__ == '__main__':
    test_redist()
