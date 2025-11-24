import pytest

from gpaw.core import PWDesc, UGDesc
from gpaw.core.matrix import Matrix
from gpaw.mpi import world
import math

def comms(comm):
    """Yield communicator combinations."""
    for s in [1, 2, 4, 8]:
        if s > comm.size:
            return
        domain_comm = comm.new_communicator(
            range(comm.rank // s * s, comm.rank // s * s + s))
        band_comm = comm.new_communicator(
            range(comm.rank % s, comm.size, s))
        yield domain_comm, band_comm


def func(f):
    """Operator for matrix elements."""
    g = f.copy()
    g.data *= 2.3
    return g


# TODO: test also UGArray
@pytest.mark.parametrize('db_index', range(world.size.bit_length()))
@pytest.mark.parametrize('dtype', [float, complex])
@pytest.mark.parametrize('nbands', [1, 7, 21])
@pytest.mark.parametrize('function', [None, func])
def test_me(comm, db_index, dtype, nbands, function):
    domain_comm, band_comm = list(comms(comm))[db_index]
    a = 2.5
    n = 20
    grid = UGDesc(cell=[a, a, a], size=(n, n, n))
    desc = PWDesc(ecut=50, cell=grid.cell)
    desc = desc.new(comm=domain_comm, dtype=dtype)
    f = desc.empty(nbands, comm=band_comm)
    f.randomize()

    M = f.matrix_elements(f, function=function)
    out = Matrix(nbands, nbands, dist=(band_comm, -1, 1), dtype=dtype)
    out.data[:] = 1e308  # will overflow when multiplied by 2
    f.matrix_elements(f, function=function, out=out)

    f1 = f.gathergather()
    M2 = M.gather()
    if f1 is not None:
        M1 = f1.matrix_elements(f1, function=function)
        M1.tril2full()
        M2.tril2full()
        dM = M1.data - M2.data
        assert abs(dM).max() < 1e-11

    if function is None:
        g = f.new()
        g.randomize()
        M = f.matrix_elements(g)

        f1 = f.gathergather()
        g1 = g.gathergather()
        M2 = M.gather()
        if f1 is not None:
            M1 = f1.matrix_elements(g1)
            M1.tril2full()
            M2.tril2full()
            dM = M1.data - M2.data
            assert abs(dM).max() < 1e-11


if __name__ == '__main__':
    d, b = list(comms())[0]
    test_me(d, b, float, 4, None)
