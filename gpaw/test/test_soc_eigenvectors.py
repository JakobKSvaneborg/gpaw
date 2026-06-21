import numpy as np
import pytest

from gpaw.spinorbit import soc_eigenstates


@pytest.mark.soc
def test_soc_eigenvectors_parallel_collect(gpw_files, mpi):
    """Regression test for parallel collection of SOC eigenvectors.

    On master, ``BZWaveFunctions.eigenvectors()`` crashed when run in MPI
    parallel with ``kpt_comm.size > 1``: ``WaveFunction.v_mn`` was stored as
    a transposed view (``v_nm.T``) and therefore not C-contiguous, which the
    GPAW MPI ``send`` rejects with::

        TypeError: Not a proper NumPy array: needs to be c-contiguous,
                   aligned and writeable.

    Loading the gpw-file with the world communicator is what makes
    ``kpt_comm.size > 1`` (and exercises the failure) when the test is run
    on more than one rank.
    """
    calc = mpi.GPAW(gpw_files['mos2_pw'])
    soc = soc_eigenstates(calc)

    # This call crashed on master in parallel:
    v_kmn = soc.eigenvectors()

    nbzkpts, nbands = soc.shape
    assert v_kmn.shape == (nbzkpts, nbands, nbands)
    assert v_kmn.dtype == complex
    assert v_kmn.flags['C_CONTIGUOUS']

    # The eigenvectors should be unitary (rows of v_mn are orthonormal):
    eye = np.eye(nbands)
    for v_mn in v_kmn:
        assert np.allclose(v_mn @ v_mn.conj().T, eye, atol=1e-6)
