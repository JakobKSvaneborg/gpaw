import pytest
from gpaw.gpu.mpi import CuPyMPI
from gpaw.new.c import GPU_AWARE_MPI
from gpaw.core import UGDesc

# We allow the world import due to special parametrization of parallel tests
from gpaw.mpi import world


parametrization = {1: [1], 2: [1, 2], 4: [1, 2, 4], 8: [1, 2, 4, 8]}


@pytest.fixture(params=parametrization[world.size])
def domain_band_comms(request, comm):
    """Generate parametrization of a 2-tuple of domain and band comms

    The band comm size will go in powers of two from 1 to world.size,
    and domain comm will be the reciprocal communicator.
    """
    s = request.param
    domain_comm = comm.new_communicator(
        range(comm.rank // s * s, comm.rank // s * s + s))
    band_comm = comm.new_communicator(
        range(comm.rank % s, comm.size, s))
    return domain_comm, band_comm


@pytest.fixture(params=range(5))
def grid(comm, request):
    if not GPU_AWARE_MPI:
        comm = CuPyMPI(comm)
    a = 1.0
    decomp = {1: [[0, 4], [0, 4], [0, 4]],
              2: [[0, 2, 4], [0, 4], [0, 4]],
              4: [[0, 2, 4], [0, 2, 4], [0, 4]],
              8: [[0, 1, 2, 3, 4], [0, 2, 4], [0, 4]]}[comm.size]
    grid = UGDesc(cell=[a, a, a], size=(4, 4, 4), comm=comm, decomp=decomp)
    gridc = grid.new(dtype=complex)

    g1 = grid.empty()
    g1.data[:] = 1.0

    g2 = grid.empty()
    g2.data[:] = 1.0
    g2.data += [0, 1, 0, -1]

    g3 = gridc.empty()
    g3.data[:] = 1.0

    g4 = gridc.empty()
    g4.data[:] = 1.0
    g4.data += [0, 1, 0, -1]

    g5 = gridc.empty()
    g5.data[:] = 1.0 + 1.0j
    g5.data += [0, 1, 0, -1]

    return [g1, g2, g3, g4, g5][request.param]
