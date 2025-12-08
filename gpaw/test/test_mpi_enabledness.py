"""Test that MPI is initialized if and only if we run "gpaw -P<n> python."""

import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def skip_if_mpi_initialized(comm):
    if 'mpi4py.MPI' in sys.modules:
        pytest.fail('Someone initialized MPI via mpi4py')

    from gpaw.mpi import SerialCommunicator

    if not isinstance(comm, SerialCommunicator):
        pytest.skip('Cannot create non-MPI process from MPI process')


def test_ordinary_python():
    assert subprocess.call([sys.executable, __file__]) == MPI_MISSING


def test_gpaw_python():
    # Note: gpaw -P1 does not actually enable MPI.  Is that the best?
    assert subprocess.call([sys.executable, '-m', 'gpaw', '-P1',
                            'python', __file__]) == MPI_PRESENT


MPI_MISSING = 234
MPI_PRESENT = 235


if __name__ == '__main__':
    # MPI was called if we did it in _broadcast_imports,
    # or if someone initialized mpi4py (which we don't know unless we did it,
    # but if we did it, it was also in _broadcast_imports.
    #
    # We cannot completely guarantee that nobody created a _gpaw.Communicator()
    # which is what actually causes GPAW to initialize MPI, but this should
    # be the best point to intercept the initialization.
    from gpaw._broadcast_imports import world

    if world is None:
        sys.exit(MPI_MISSING)
    else:
        sys.exit(MPI_PRESENT)
