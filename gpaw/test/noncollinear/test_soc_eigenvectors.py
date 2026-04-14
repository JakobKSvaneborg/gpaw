import numpy as np
import pytest

from gpaw.spinorbit import soc_eigenstates


@pytest.mark.soc
def test_soc_eigenvectors_parallel_collect(gpw_files, mpi):
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
