import numpy as np
import pytest

from gpaw.core.matrix import Matrix
from gpaw.mpi import world


@pytest.mark.skipif(world.size != 4, reason='size!=4')
def test_matrix_eigh(rng):
    """Test eigenvalues for different BLACS layouts.

    See also #269.
    """
    N = 6
    x = 0.01

    A0 = Matrix(N, N, dist=(world, 1, 1), dtype=complex)

    if world.rank == 0:
        A0.data[:] = np.diag(np.arange(N) + 1)
        A0.data += rng.uniform(-x, x, (N, N))
        A0.data += A0.data.conj().T
        B = Matrix(N, N, data=A0.data.copy())
        eigs0 = B.eigh(cc=True)
    else:
        eigs0 = np.empty(N)

    world.broadcast(eigs0, 0)

    A = Matrix(N, N, dist=(world, 2, 2, 2), dtype=complex)
    B0 = Matrix(N, N, dist=(world, 1, 1), dtype=complex)

    for dist in [(1, 2), (2, 1), (2, 2)]:
        A0.redist(A)
        eigs = A.eigh(cc=True, scalapack=(world, *dist, 2))
        assert eigs == pytest.approx(eigs0, abs=1e-13)
        print(world.rank, eigs)
        A.redist(B0)
        if world.rank == -1:
            assert abs(B0.data) == pytest.approx(abs(B.data), abs=1e-13)
