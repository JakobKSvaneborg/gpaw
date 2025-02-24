import gpaw.cgpaw as cgpaw
from gpaw.gpu import cupy as cp, cupy_is_fake
import numpy as np
from gpaw.gpu.cpupy import asnumpy

def eigh_magma_cpu(matrix: np.ndarray, UPLO: str) -> tuple[np.ndarray, np.ndarray]:
    """"""
    assert cgpaw.have_magma, "Must compile with MAGMA support"

    if matrix.dtype == np.complex128:
        eigvals, eigvects = cgpaw.eigh_magma_zheevd(matrix, UPLO)

    elif matrix.dtype == np.float64:
        eigvals, eigvects = cgpaw.eigh_magma_dsyevd(matrix, UPLO)

    else:
        raise TypeError("Unsupported matrix dtype")

    # MAGMA eigenvectors are on rows, numpy/cupy has them on columns
    return eigvals, eigvects.T


def eigh_magma_gpu(matrix: cp.ndarray, UPLO: str) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Returns
    -------
    w : (N,) ndarray
        Eigenvalues in ascending order
    v : (N, N) ndarray
        Matrix containing orthonormal eigenvectors.
        Vector corresponding to ``w[i]`` is in column ``v[:,i]``.

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
    eigvals = cp.empty((matrix.shape[0],), dtype=np.float64)

    if matrix.dtype == np.complex128:
        cgpaw.eigh_magma_zheevd_gpu(matrix,
                                    UPLO,
                                    eigvals,
                                    eigvects)

    elif matrix.dtype == np.float64:
        cgpaw.eigh_magma_dsyevd_gpu(matrix,
                                    UPLO,
                                    eigvals,
                                    eigvects)

    else:
        raise TypeError("Unsupported matrix dtype")

    # MAGMA eigenvectors are on rows, numpy/cupy has them on columns
    return eigvals, eigvects.T