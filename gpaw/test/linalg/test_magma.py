"""Tests for MAGMA eigensolver wrappers"""
import numpy as np
import pytest
from gpaw.new.magma import eigh_magma_cpu, eigh_magma_gpu
from gpaw.cgpaw import have_magma
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake

# Comparing eigenvectors from different solvers is challenging because of
# phase ambiguity. One solution would be to fix the phase according to some
# sensible convention; however this doesn't seem to work very well at single
# precision. Instead, we check that the eigenvalues, eigenvector pairs
# (lam, v) satisfy A.v == lam*v, and assert that eigenvalues from Magma
# match with those from Numpy/Cupy


def assert_eigenpairs(A, eigvals, eigvecs, rtol=1e-12, atol=1e-12) -> None:
    """Checks that A @ v == lam*v and asserts on failure."""

    xp = cp if isinstance(A, cp.ndarray) else np

    for i in range(eigvecs.shape[1]):
        v = eigvecs[:, i]
        lam = eigvals[i]
        lhs = A @ v
        rhs = lam * v

        xp.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)
    #


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
            A = (
                rng.random((n, n), dtype=dtype_real)
                + 1j * rng.random((n, n), dtype=dtype_real)
            )
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
    eigvals, eigvecs = eigh_magma_cpu(arr, uplo)

    eigvals_np, eigvecs_np = np.linalg.eigh(arr, UPLO=uplo)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-6
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    np.testing.assert_allclose(eigvals, eigvals_np, atol=atol)
    assert_eigenpairs(arr, eigvals, eigvecs, rtol=rtol, atol=atol)


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
    """Compare eigh output of CUPY and MAGMA (GPU)"""

    arr = eigh_test_matrix(matrix_size, dtype=dtype, backend='cupy')

    eigvals, eigvecs = eigh_magma_gpu(arr, uplo)
    eigvals_cp, eigvecs_cp = cp.linalg.eigh(arr, UPLO=uplo)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    cp.testing.assert_allclose(eigvals, eigvals_cp, atol=atol, rtol=rtol)
    assert_eigenpairs(arr, eigvals, eigvecs, rtol=rtol, atol=atol)
