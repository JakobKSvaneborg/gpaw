"""Test that collect_A_SS works with uneven k-point distribution.
"""
import numpy as np
import pytest

from gpaw.response.bse import BSE


class MockContext:
    def __init__(self, comm):
        self.comm = comm


class MockKD:
    def __init__(self, nK):
        self.nbzkpts = nK


class MockBSE:
    """Stub with just enough attributes for BSE.collect_A_SS to work."""

    def __init__(self, nK, nv, nc, comm):
        self.context = MockContext(comm)
        self.kd = MockKD(nK)
        self.nv = nv
        self.nc = nc
        self.nK = nK
        self.nS = nK * nv * nc
        self.ns = -(-nK // comm.size) * nv * nc  # padded size

    # Bind the real BSE methods to this stub
    collect_A_SS = BSE.collect_A_SS
    parallelisation_kpoints = BSE.parallelisation_kpoints


@pytest.mark.response
def test_collect_A_SS_uneven(mpi):
    """collect_A_SS must handle uneven k-point distribution across ranks."""
    comm = mpi.comm
    if comm.size < 2:
        pytest.skip('Need at least 2 MPI ranks')

    # Choose nK that does NOT divide evenly by comm.size
    nK = comm.size * 5 + 1  # guaranteed uneven
    nv, nc = 2, 2
    nS = nK * nv * nc

    stub = MockBSE(nK, nv, nc, comm)
    _, myKsize = stub.parallelisation_kpoints()
    mySsize = myKsize * nv * nc

    # Build a known global matrix: A_SS[i, j] = i * nS + j
    # Each rank owns rows [offset : offset + mySsize]
    offset = 0
    for r in range(comm.rank):
        _, ksize = stub.parallelisation_kpoints(r)
        offset += ksize * nv * nc

    A_sS = np.empty((mySsize, nS), dtype=complex)
    for i in range(mySsize):
        for j in range(nS):
            A_sS[i, j] = (offset + i) * nS + j

    # Check that the correct matrix is returned
    A_SS = stub.collect_A_SS(A_sS)
    if comm.rank == 0:
        expected = np.empty((nS, nS), dtype=complex)
        for i in range(nS):
            for j in range(nS):
                expected[i, j] = i * nS + j
        assert np.allclose(A_SS, expected), (
            'Fixed collect_A_SS returned incorrect values')
