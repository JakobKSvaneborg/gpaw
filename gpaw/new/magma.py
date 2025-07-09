import gpaw.cgpaw as cgpaw
from gpaw.gpu import cupy as cp, cupy_is_fake
import numpy as np
from gpaw.gpu.cpupy import asnumpy
from gpaw.utilities import as_real_dtype


def eigh_magma_cpu(matrix: np.ndarray, UPLO: str) -> tuple[np.ndarray,
                                                           np.ndarray]:
    """
    Wrapper for MAGMA symmetric/Hermitian eigensolvers, CPU version.

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
