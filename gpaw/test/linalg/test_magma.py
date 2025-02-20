"""Tests for MAGMA eigensolver wrappers"""
import numpy as np
import pytest
from gpaw.new.magma import eigh_magma_cpu, eigh_magma_gpu
from gpaw.cgpaw import have_magma
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake # GPU tests don't work with fake cupy

def fix_eigenvector_phase(inout_arr):
    """Helper function for comparing eigenvector output from different
    solvers. Rotates the input eigenvector matrix so that the first column
    of each row becomes a real non-negative number.
    Input is modified in-place.
    """
    assert inout_arr.ndim == 2

    # Works for cupy arrays too because the dtypes are compatible
    if np.issubdtype(inout_arr.dtype, np.floating):
        # Real matrices
        for i in range(inout_arr.shape[0]):
            if inout_arr[i, 0] < 0:
                inout_arr[i] *= -1
        return inout_arr


@pytest.fixture
def symmetric_matrix():
    def _generate(n: int, backend: str ='numpy', seed: int = 42):
        assert backend in ['numpy', 'cupy']

        if backend == 'cupy':
            xp = cp
        else:
            xp = np

        rng = xp.random.default_rng(seed)
        A = rng.random((n, n))
        return (A + A.T) / 2

    return _generate

@pytest.mark.skipif(not have_magma, reason="No MAGMA")
@pytest.mark.parametrize("matrix_size, uplo", [(2, 'L'), (4, 'U')])
def test_eigh_magma_cpu(symmetric_matrix: np.ndarray,
                        matrix_size:
                        int, uplo: str) -> None:
    """Compare eigh output of Numpy and MAGMA"""

    arr = symmetric_matrix(matrix_size, backend='numpy')
    eigvals, eigvects = eigh_magma_cpu(arr, uplo)

    eigvals_np, eigvects_np = np.linalg.eigh(arr, UPLO=uplo)

    np.testing.assert_allclose(eigvals, eigvals_np, atol=1e-16)
    np.testing.assert_allclose(eigvects, eigvects_np, atol=1e-16)


@pytest.mark.skipif(not have_magma, reason="No MAGMA")
@pytest.mark.skipif(cupy_is_fake, reason="MAGMA GPU tests broken for fake cupy")
@pytest.mark.gpu
@pytest.mark.parametrize("matrix_size, uplo", [(2, 'L'), (4, 'U')])
def test_eigh_magma_gpu(symmetric_matrix: cp.ndarray,
                        matrix_size: int,
                        uplo: str):
    """Compare eigh output of CUPY and MAGMA"""

    arr = symmetric_matrix(matrix_size, backend='cupy')
    eigvals, eigvects = eigh_magma_gpu(arr, uplo)

    eigvals_np, eigvects_np = cp.linalg.eigh(arr, UPLO=uplo)

    cp.testing.assert_allclose(eigvals, eigvals_np, atol=1e-16)
    cp.testing.assert_allclose(eigvects, eigvects_np, atol=1e-16)
