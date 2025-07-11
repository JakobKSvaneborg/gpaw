# Tests of GPU functionality

import numpy as np
from gpaw.gpu import cupy as cp

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
