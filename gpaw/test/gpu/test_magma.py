"""Tests the magma bindings directly.
See also gpaw/test/gpu/test_diagonalizers.py"""

import pytest
from gpaw.cgpaw import have_magma

if not have_magma:
    pytest.skip("No MAGMA", allow_module_level=True)

import gpaw.cgpaw as cgpaw
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.utilities import as_real_dtype
from gpaw.test.gpu import assert_eigenpairs, fill_uplo
from gpaw.test.gpu.test_diagonalizers import fixt_eigh_test_matrix
import numpy as np
import sys

# if cupy_is_fake:
#     pytest.skip("Not testing MAGMA with fake Cupy", allow_module_level=True)

# Hack around cgpaw not being a package: can't do `from cgpaw.gpu import magma`.
# Will fix in a followup MR
magma = cgpaw.gpu.magma

@pytest.mark.parametrize("matrix_dtype", [np.float64, np.complex128])
def test_invalid_input(matrix_dtype: np.dtype):
    """Checks that we correctly catch non-conforming inputs.
    Arrays with non-contiguous memory layout are especially dangerous so the
    function must error out on them.
    """
    matrix = np.zeros((2, 2), dtype=matrix_dtype)
    eigvals = np.zeros(2)

    # test invalid uplo
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix, eigvals, "O", 1)

    # test wrong eigenvalue array shape
    eigvals_bad = np.zeros(1)
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix, eigvals_bad, "U", 1)

    # test non-C-contiguous input matrix
    matrix_bad = np.zeros_like(matrix, order='F')
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix_bad, eigvals, "U", 1)

    # test strided eigenvalue array (non-contiguous 1D)
    eigvals_bad = np.arange(4, dtype=np.float64)[::2]
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix, eigvals_bad, "U", 1)

    # test dtype mismatch
    eigvals_bad = np.zeros_like(eigvals, dtype=np.float32)
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix, eigvals_bad, "U", 1)

    # test non-supported matrix dtypes
    matrix_bad = np.zeros_like(matrix, dtype=np.int64)
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix_bad, eigvals, "U", 1)

    matrix_bad = np.zeros_like(matrix, dtype=np.float16)
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix_bad, eigvals, "U", 1)

    # Test invalid gpu request
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix, eigvals, "U", 0)
    with pytest.raises(ValueError):
        magma.eigh_magma_numpy(matrix, eigvals, "U", sys.maxsize)


@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                          np.complex64, np.complex128])
@pytest.mark.parametrize("uplo", ['U', "L"])
@pytest.mark.parametrize("xp", [np, cp])
def test_magma_eigh(fixt_eigh_test_matrix,
                    dtype: np.dtype,
                    uplo: str,
                    xp):
    """"""
    matrix_size = 3
    matrix_orig: np.ndarray | cp.ndarray = fixt_eigh_test_matrix(
        matrix_size,
        dtype=dtype,
        backend='cupy' if xp is cp else 'numpy'
    )
    eigvals = xp.empty(matrix_size, dtype=as_real_dtype(dtype))

    # reference
    eigvals_ref, eigvecs_ref = xp.linalg.eigh(matrix_orig, uplo)

    matrix = matrix_orig.copy()
    if xp is np:
        magma.eigh_magma_numpy(matrix, eigvals, uplo, 1)
    else:
        magma.eigh_magma_cupy(matrix, eigvals, uplo)

    eigvecs = matrix.conj().T

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    # matrix may have not been symmetric/hermitian (intentionally, for uplo test)
    true_matrix = fill_uplo(matrix_orig, uplo)

    xp.testing.assert_allclose(eigvals, eigvals_ref, rtol=rtol, atol=atol)
    assert_eigenpairs(true_matrix, eigvals, eigvecs, rtol=rtol, atol=atol)
