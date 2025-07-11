
import numpy as np
import pytest
from gpaw.gpu import cupy as cp

@pytest.fixture(scope="function")
def fixt_raw_hermitian_matrix():
    """Symmetric if dtype is real, Hermitian otherwise.
    This is a 'raw' numpy or cupy array, NOT gpaw Matrix object.
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
