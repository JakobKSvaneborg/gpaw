import gpaw.cgpaw as cgpaw
from gpaw.gpu import cupy as cp, cupy_is_fake
import numpy as np
from gpaw.gpu.cpupy import asnumpy
from gpaw.utilities import as_real_dtype


def eigh_magma_cpu(matrix: np.ndarray, UPLO: str) -> tuple[np.ndarray,
                                                           np.ndarray]:
    """
    Wrapper for MAGMA symmetric/Hermitian eigensolvers, CPU version.
    Same conventions for input/output as in numpy.linalg.eigh().

    Parameters
    ----------
    matrix : (N, N) numpy.ndarray
        The matrix to diagonalize. Must be symmetric or Hermitian.
    UPLO : str
        Whether the upper or lower part of the matrix is stored.
        Choose 'U' or 'L'.

    Returns
    -------
    w : (N,) numpy.ndarray
        Eigenvalues in ascending order
    v : (N, N) numpy.ndarray
        Matrix containing orthonormal eigenvectors.
        Eigenvector corresponding to ``w[i]`` is in column ``v[:,i]``.
    """

    assert cgpaw.have_magma, "Must compile with MAGMA support"

    # The internal function handles output array creation
    eigvals, eigvects = cgpaw._eigh_magma_cpu(matrix, UPLO)

    # MAGMA eigenvectors are on rows, numpy/cupy has them on columns
    return eigvals, np.conjugate(eigvects).T


def eigh_magma_gpu(matrix: cp.ndarray, UPLO: str) -> tuple[cp.ndarray,
                                                           cp.ndarray]:
    """
    Wrapper for MAGMA symmetric/Hermitian eigensolvers, GPU version.
    Same conventions for input/output as in numpy.linalg.eigh().

    Parameters
    ----------
    matrix : (N, N) cupy.ndarray
        The matrix to diagonalize. Must be symmetric or Hermitian.
    UPLO : str
        Whether the upper or lower part of the matrix is stored.
        Choose 'U' or 'L'.

    Returns
    -------
    w : (N,) cupy.ndarray
        Eigenvalues in ascending order
    v : (N, N) cupy.ndarray
        Matrix containing orthonormal eigenvectors.
        Eigenvector corresponding to ``w[i]`` is in column ``v[:,i]``.
    """
    assert cgpaw.have_magma, "Must compile with MAGMA support"

    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

    if cupy_is_fake:
        eigval_np, eigvect_np = eigh_magma_cpu(asnumpy(matrix), UPLO)
        return cp.asarray(eigval_np), cp.asarray(eigvect_np)

    # Alloc output arrays with CUPY.
    # Necessary because the C code has no easy access to CUPY array creation
    eigvects = cp.empty_like(matrix)

    # Only symmetric/Hermitian matrices supported for now,
    # so eigenvalues are always real
    eigval_dtype = as_real_dtype(matrix.dtype)

    eigvals = cp.empty((matrix.shape[0]), dtype=eigval_dtype)

    # Will throw if matrix dtype is unsupported
    cgpaw._eigh_magma_gpu(matrix, UPLO, eigvals, eigvects)

    # MAGMA eigenvectors are on rows, numpy/cupy has them on columns
    return eigvals, cp.conjugate(eigvects).T
