"""Test that collect_A_SS works with uneven k-point distribution.

When the number of k-points does not divide evenly by the number of
MPI ranks, the last rank has fewer rows. The original collect_A_SS
used the padded self.ns (same on all ranks) for receive buffers,
which caused a ValueError on uneven distributions.

This test must be run with at least 2 MPI ranks and an odd number of
k-points to exercise the bug.
"""
import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.response.bse import BSE


class MockContext:
    def __init__(self, comm):
        self.comm = comm


class MockKD:
    def __init__(self, nK):
        self.nbzkpts = nK


class CollectStub:
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


def _collect_A_SS_original(stub, A_sS):
    """Original (buggy) implementation using padded self.ns."""
    comm = stub.context.comm
    if comm.rank == 0:
        A_SS = np.zeros((stub.nS, stub.nS), dtype=complex)
        A_SS[:len(A_sS)] = A_sS
        Ntot = len(A_sS)
        for rank in range(1, comm.size):
            buf = np.empty((stub.ns, stub.nS), dtype=complex)
            comm.receive(buf, rank, tag=123)
            A_SS[Ntot:Ntot + stub.ns] = buf
            Ntot += stub.ns
    else:
        comm.send(A_sS, 0, tag=123)
    comm.barrier()
    if comm.rank == 0:
        return A_SS


@pytest.mark.response
def test_collect_A_SS_uneven():
    """collect_A_SS must handle uneven k-point distribution across ranks."""
    comm = world
    if comm.size < 2:
        pytest.skip('Need at least 2 MPI ranks')

    # Choose nK that does NOT divide evenly by comm.size
    nK = comm.size * 5 + 1  # guaranteed uneven
    nv, nc = 2, 2
    nS = nK * nv * nc

    stub = CollectStub(nK, nv, nc, comm)
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

    # --- The original code should fail on uneven distributions ---
    original_failed = False
    try:
        _collect_A_SS_original(stub, A_sS)
    except (ValueError, Exception):
        original_failed = True
    # Synchronise: all ranks must agree on whether it failed.
    # (The error occurs on rank 0; other ranks just did send+barrier.)
    failed_flag = np.array([1.0 if original_failed else 0.0])
    comm.max(failed_flag)
    original_failed = failed_flag[0] > 0.5

    assert original_failed, (
        'Expected the original collect_A_SS to fail with uneven '
        f'k-point distribution (nK={nK}, comm.size={comm.size})')

    # --- The fixed code should succeed and return the correct matrix ---
    A_SS = stub.collect_A_SS(A_sS)
    if comm.rank == 0:
        expected = np.empty((nS, nS), dtype=complex)
        for i in range(nS):
            for j in range(nS):
                expected[i, j] = i * nS + j
        assert np.allclose(A_SS, expected), (
            'Fixed collect_A_SS returned incorrect values')
