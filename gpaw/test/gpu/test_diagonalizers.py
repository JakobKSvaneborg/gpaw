import numpy as np
import pytest
from typing import TYPE_CHECKING, Type
from gpaw.core.matrix import Matrix
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.gpu.diagonalization import CPUPYDiagonalizer, CuPyDiagonalizer, DiagonalizerOptions
from gpaw.gpu.diagonalization.magma_diagonalizer import MagmaDiagonalizer
from gpaw.test.gpu import assert_eigenpairs

if TYPE_CHECKING:
    from gpaw.gpu.diagonalization import GPUDiagonalizer


@pytest.mark.gpu
# @pytest.mark.parametrize("matrix_size, dtype, uplo, inplace",
#                          [(4, np.float32, 'L', False),
#                           (8, np.complex128, 'U', False),
#                           (8, np.complex128, 'L', True)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("matrix_size", [4])
@pytest.mark.parametrize("uplo", ['L', 'U'])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("diagonalizer_class",
                         [CPUPYDiagonalizer,
                          CuPyDiagonalizer,
                          MagmaDiagonalizer])
def test_matrix_diagonalizer(fixt_raw_hermitian_matrix: cp.ndarray,
               diagonalizer_class: Type["GPUDiagonalizer"],
               matrix_size: int,
               dtype: np.dtype,
               uplo: str,
               inplace: bool):
    """"""

    if cupy_is_fake and diagonalizer_class is not CPUPYDiagonalizer:
        pytest.skip("CuPy is fake")
    elif not cupy_is_fake and diagonalizer_class is CPUPYDiagonalizer:
        pytest.skip("Not testing cpupy when running with real CuPy")

    raw_matrix = fixt_raw_hermitian_matrix(matrix_size, dtype=dtype, backend='cupy')
    matrix = Matrix(matrix_size, matrix_size, dtype=dtype, data=raw_matrix)

    #matrix_orig = matrix.copy()
    raw_matrix_orig = cp.copy(raw_matrix)

    # Reference values
    eigvals_cp, eigvecs_cp = cp.linalg.eigh(raw_matrix, UPLO=uplo)

    diagonalizer = diagonalizer_class()
    options = DiagonalizerOptions(uplo=uplo, inplace=inplace)

    eigvals, eigvecs = diagonalizer.eigh(matrix, options)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    if options.inplace:
        assert eigvecs is matrix

    # Endless struggle with fake cupy
    if cupy_is_fake:
        eigvals, eigvecs_raw = cp.asnumpy(eigvals), cp.asnumpy(eigvecs.data)
        eigvals_ref = cp.asnumpy(eigvals_cp)
        raw_matrix_orig = cp.asnumpy(raw_matrix_orig)
        xp = np
    else:
        xp = cp
        eigvecs_raw = eigvecs.data
        eigvals_ref = eigvals_cp

    xp.testing.assert_allclose(eigvals, eigvals_ref, atol=atol, rtol=rtol)

    # TODO: use Matrix.multiply and other direct Matrix routines.
    # The function below operates on raw xp.ndarrays, and we need to transpose
    # eigvecs back to original convention => very confusing
    assert_eigenpairs(raw_matrix_orig, eigvals, eigvecs_raw.T, rtol=rtol, atol=atol)
