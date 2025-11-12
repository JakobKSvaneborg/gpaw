import numpy as np
import pytest
from gpaw.gpu import cupy as cp


@pytest.fixture(scope="function")
def fixt_eigh_test_matrix():
    """Generates a random n-by-n matrix. NOT symmetric/hermitian:
    eigh() solvers read only upper/lower half of the matrix, so by passing
    a non-Hermitian matrix we can test that our wrappers correctly interpret
    the input (ie. test the 'uplo' argument).
    """
    def _generate(n: int, dtype: np.dtype = np.float64,
                  backend: str = 'numpy', seed: int = 42):

        assert backend in ['numpy', 'cupy']

        if backend == 'cupy':
            xp = cp
        else:
            xp = np

        rng = xp.random.default_rng(seed)

        if not np.issubdtype(dtype, np.complexfloating):
            A = rng.random((n, n), dtype=dtype)
            return A
        else:
            # Only 32/64 bit precision implemented
            assert dtype == np.complex64 or dtype == np.complex128
            dtype_real = np.float32 if dtype == np.complex64 else np.float64
            A = (
                rng.random((n, n), dtype=dtype_real)
                + 1j * rng.random((n, n), dtype=dtype_real)
            )
            # Set imaginary parts to zero on the diagonal. Most Hermitian
            # solvers seem to ignore them, but at least old Cupy-11 gives
            # different eigenvalues if the diagonal is not real.
            xp.fill_diagonal(A.imag, 0)

            return A

    return _generate
