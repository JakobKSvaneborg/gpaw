import gpaw.cgpaw as cgpaw
from gpaw.gpu import cupy as cp
import numpy as np
from gpaw.gpu import cupy_is_fake
from gpaw.gpu.cpupy import asnumpy

def eigh_magma_cpu(matrix: np.ndarray, UPLO: str) -> tuple[np.ndarray, np.ndarray]:
    """"""
    assert cgpaw.have_magma, "Must compile with MAGMA support"

    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise NotImplementedError

    elif np.issubdtype(matrix.dtype, np.floating):
        eigvals, eigvects = cgpaw.eigh_magma_dsyevd(matrix, UPLO)

    else:
        ## Not floating point
        raise NotImplementedError

    # Change eigenvectors to Numpy convention
    return eigvals, eigvects.T


def eigh_magma_gpu(matrix: cp.ndarray, UPLO: str) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Returns
    -------
    w : (N,) ndarray
        Eigenvalues in ascending order
    v : (N, N) ndarray
        Matrix containing orthonormal eigenvectors.
        Vector corresponding to ``w[i]`` is in ``v[:,i]``.

    """
    assert cgpaw.have_magma, "Must compile with MAGMA support"

    if cupy_is_fake:
        return eigh_magma_cpu(asnumpy(matrix), UPLO)

    if np.issubdtype(matrix.dtype, np.complexfloating):
        raise NotImplementedError

    elif np.issubdtype(matrix.dtype, np.floating):
        eigvals, eigvects = cgpaw.eigh_magma_dsyevd(matrix, UPLO)

    else:
        ## Not floating point
        raise NotImplementedError

    # Change eigenvectors to Numpy convention
    return eigvals, eigvects.T