"""Tests for MAGMA eigensolver wrappers"""
import numpy as np
import pytest
from gpaw.new.magma import eigh_magma_cpu, eigh_magma_gpu
from gpaw.cgpaw import have_magma
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake


def fix_eigenvector_phase(inout_arr):
    """Helper function for comparing eigenvector output from different
    solvers. Rotates eigenvectors in the input matrix so that the first
    element of each vector is real and non-negative.
    Input is modified in-place.
    NB: eigenvectors are on columns.
    """
    assert inout_arr.ndim == 2
    # Works for cupy arrays too because the dtypes are compatible

    if np.issubdtype(inout_arr.dtype, np.complexfloating):
        # Complex matrices
        for i in range(inout_arr.shape[1]):
            phase = np.angle(inout_arr[0, i])
            if phase != 0:
                rotation = np.exp(phase * (-1j))
                inout_arr[:, i] *= rotation

    elif np.issubdtype(inout_arr.dtype, np.floating):
        # Real matrices
        for i in range(inout_arr.shape[1]):
            if inout_arr[0, i] < 0:
                inout_arr[:, i] *= -1

    return inout_arr


@pytest.fixture
def eigh_test_matrix():
    """Symmetric if dtype is real, Hermitian otherwise."""
    def _generate(n: int, dtype: np.dtype = np.float64,
                  backend: str = 'numpy', seed: int = 42):

        assert backend in ['numpy', 'cupy']

        if backend == 'cupy':
            xp = cp
        else:
            xp = np

        rng = xp.random.default_rng(seed)

        if not np.issubdtype(dtype, np.complexfloating):
            # Real dtype, return symmetric matrix
            A = rng.random((n, n), dtype=dtype)
            return (A + A.T) / 2

        else:
            # Only 32/64 bit precision implemented
            assert dtype == np.complex64 or dtype == np.complex128
            dtype_real = np.float32 if dtype == np.complex64 else np.float64
            # Create Hermitian matrix
            A = ( rng.random((n, n), dtype=dtype_real)
                + 1j * rng.random((n, n), dtype=dtype_real) )
            return (A + A.T.conj()) / 2

    return _generate


@pytest.mark.skipif(not have_magma, reason="No MAGMA")
@pytest.mark.parametrize("matrix_size, dtype, uplo",
                         [(2, np.float32, 'L'),
                          (3, np.float64, 'U'),
                          (2, np.complex64, 'U'),
                          (4, np.complex128, 'L')])
def test_eigh_magma_cpu(eigh_test_matrix: np.ndarray,
                        matrix_size: int,
                        dtype: np.dtype,
                        uplo: str) -> None:
    """Compare eigh output of Numpy and MAGMA"""

    arr = eigh_test_matrix(matrix_size, dtype=dtype, backend='numpy')
    eigvals, eigvects = eigh_magma_cpu(arr, uplo)

    eigvals_np, eigvects_np = np.linalg.eigh(arr, UPLO=uplo)

    fix_eigenvector_phase(eigvects)
    fix_eigenvector_phase(eigvects_np)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-6

    np.testing.assert_allclose(eigvals, eigvals_np, atol=atol)
    np.testing.assert_allclose(eigvects, eigvects_np, atol=atol)


# MAGMA seems to do small matrices (N <= 128) on the CPU.
# So need a large matrix for honest GPU tests
@pytest.mark.skipif(not have_magma, reason="No MAGMA")
@pytest.mark.skipif(cupy_is_fake,
                    reason="MAGMA GPU tests disabled for fake cupy")
@pytest.mark.gpu
@pytest.mark.parametrize("matrix_size, dtype, uplo",
                         [(16, np.float32, 'L'),
                          (130, np.float32, 'L'),
                          (256, np.float64, 'U'),
                          (150, np.complex64, 'U'),
                          (140, np.complex128, 'L')])
def test_eigh_magma_gpu(eigh_test_matrix: cp.ndarray,
                        matrix_size: int,
                        dtype: np.dtype,
                        uplo: str):
    """Compare eigh output of CUPY and MAGMA"""

    arr = eigh_test_matrix(matrix_size, dtype=dtype, backend='cupy')
    eigvals, eigvects = eigh_magma_gpu(arr, uplo)

    eigvals_cp, eigvects_cp = cp.linalg.eigh(arr, UPLO=uplo)

    fix_eigenvector_phase(eigvects)
    fix_eigenvector_phase(eigvects_cp)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-6

    cp.testing.assert_allclose(eigvals, eigvals_cp, atol=atol)
    cp.testing.assert_allclose(eigvects, eigvects_cp, atol=atol)