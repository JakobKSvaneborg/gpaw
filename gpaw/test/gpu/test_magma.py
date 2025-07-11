"""Tests for MAGMA eigensolver wrappers"""
import numpy as np
import pytest
from gpaw.new.magma import eigh_magma_cpu
from gpaw.gpu.diagonalization.magma_diagonalizer import MagmaDiagonalizer
from gpaw.gpu.diagonalization.diagonalizer import DiagonalizerOptions
from gpaw.cgpaw import have_magma
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.test.gpu import assert_eigenpairs


# Comparing eigenvectors from different solvers is challenging because of
# phase ambiguity. One solution would be to fix the phase according to some
# sensible convention; however this doesn't seem to work very well at single
# precision. Instead, we check that the eigenvalues, eigenvector pairs
# (lam, v) satisfy A.v == lam*v, and assert that eigenvalues from Magma
# match with those from Numpy/Cupy.
# We also check unitarity/orthonormality of the eigenvector matrix.


@pytest.mark.skipif(not have_magma, reason="No MAGMA")
@pytest.mark.parametrize("matrix_size, dtype, uplo",
                         [(2, np.float32, 'L'),
                          (3, np.float64, 'U'),
                          (2, np.complex64, 'U'),
                          (4, np.complex128, 'L')])
def test_eigh_magma_cpu(fixt_raw_hermitian_matrix: np.ndarray,
                        matrix_size: int,
                        dtype: np.dtype,
                        uplo: str) -> None:
    """Compare eigh output of Numpy and MAGMA"""

    matrix = fixt_raw_hermitian_matrix(matrix_size, dtype=dtype, backend='numpy')
    eigvals, eigvecs = eigh_magma_cpu(matrix, uplo)

    eigvals_np, eigvecs_np = np.linalg.eigh(matrix, UPLO=uplo)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-6
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    np.testing.assert_allclose(eigvals, eigvals_np, atol=atol)
    assert_eigenpairs(matrix, eigvals, eigvecs, rtol=rtol, atol=atol)

    # check orthonormality
    np.testing.assert_allclose(np.identity(matrix_size),
                               eigvecs.conjugate().T @ eigvecs,
                               rtol=rtol, atol=atol)


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
def test_eigh_magma_gpu(fixt_raw_hermitian_matrix: cp.ndarray,
                        matrix_size: int,
                        dtype: np.dtype,
                        uplo: str):
    """Compare eigh output of CUPY and MAGMA (GPU)"""

    matrix = fixt_raw_hermitian_matrix(matrix_size, dtype=dtype, backend='cupy')

    # For checking that we don't modify this in-place
    matrix_original = cp.copy(matrix)

    diagonalizer = MagmaDiagonalizer()
    options = DiagonalizerOptions(uplo=uplo)

    eigvals, eigvecs = diagonalizer.eigh_non_distributed(matrix, options)
    eigvals_cp, eigvecs_cp = cp.linalg.eigh(matrix, UPLO=uplo)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    cp.testing.assert_allclose(eigvals, eigvals_cp, atol=atol, rtol=rtol)
    assert_eigenpairs(matrix, eigvals, eigvecs, rtol=rtol, atol=atol)

    # check orthonormality
    cp.testing.assert_allclose(cp.identity(matrix_size),
                               eigvecs.conjugate().T @ eigvecs,
                               rtol=rtol, atol=atol)

    # Check that the original matrix was preserved
    cp.testing.assert_allclose(matrix, matrix_original, rtol=rtol, atol=atol)


@pytest.mark.skipif(not have_magma, reason="No MAGMA")
@pytest.mark.skipif(cupy_is_fake,
                    reason="MAGMA GPU tests disabled for fake cupy")
@pytest.mark.gpu
@pytest.mark.parametrize("matrix_size, dtype",
                         [(4, np.float64),
                          (16, np.complex128),
                          (32, np.complex64)])
def test_eigh_magma_inplace(fixt_raw_hermitian_matrix: cp.ndarray,
                        matrix_size: int,
                        dtype: np.dtype):
    """Test the inplace option in magma eigensolvers"""

    matrix = fixt_raw_hermitian_matrix(matrix_size, dtype=dtype, backend='cupy')
    matrix_original = cp.copy(matrix)

    diagonalizer = MagmaDiagonalizer()
    options = DiagonalizerOptions(uplo='L', inplace=True)

    eigvals_cp, eigvecs_cp = cp.linalg.eigh(matrix, UPLO=options.uplo)
    eigvals, eigvecs = diagonalizer.eigh_non_distributed(matrix, options)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    cp.testing.assert_allclose(eigvals, eigvals_cp, rtol=rtol, atol=atol)
    assert_eigenpairs(matrix_original, eigvals, eigvecs, rtol=rtol, atol=atol)

    # check orthonormality
    cp.testing.assert_allclose(cp.identity(matrix_size),
                               eigvecs.conjugate().T @ eigvecs,
                               rtol=rtol, atol=atol)

    # Eigvecs should now be an alias to the input matrix
    assert matrix is eigvecs
