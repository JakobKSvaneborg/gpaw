import numpy as np


def cholesky(a):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.cholesky(a._data))


def inv(a):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(np.linalg.inv(a._data))


def eigh(a, UPLO='L'):
    from gpaw.gpu import cupy as cp
    eigvals, eigvecs = np.linalg.eigh(a._data, UPLO)
    return cp.ndarray(eigvals), cp.ndarray(eigvecs.T.copy().T)


def matrix_rank(A, tol=None, hermitian=False, *, rtol=None):
    from gpaw.gpu import cupy as cp
    return cp.ndarray(
        np.linalg.matrix_rank(A._data, tol, hermitian, rtol=rtol))
