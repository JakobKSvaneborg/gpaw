# Tests of GPU functionality

import numpy as np
from typing import Union
from gpaw.gpu import cupy as cp

# Comparing eigenvectors from different solvers is challenging because of
# phase ambiguity. One solution would be to fix the phase according to some
# sensible convention; however this doesn't seem to work very well at single
# precision. Instead, we check that the eigenvalues, eigenvector pairs
# (lam, v) satisfy A.v == lam*v, and assert that eigenvalues from Magma
# match with those from Numpy/Cupy.
# We also check unitarity/orthonormality of the eigenvector matrix.


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


def fill_uplo(matrix: Union[np.ndarray, cp.ndarray],
              from_uplo: str) -> Union[np.ndarray, cp.ndarray]:
    """Fills in lower/upper half of the input matrix so that it becomes
    Hermitian. If 'from_uplo' == 'U', fills in the lower half, and vice
    versa for 'L'."""

    xp = cp if isinstance(matrix, cp.ndarray) else np

    # Get upper or lower part only, zero elsewhere
    if from_uplo == 'U':
        m = xp.triu(matrix)
    else:
        m = xp.tril(matrix)

    return m + m.T.conj() - xp.diag(xp.real(xp.diag(m)))
