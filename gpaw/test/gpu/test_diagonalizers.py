import numpy as np
import pytest
from typing import TYPE_CHECKING, Type, Optional
from gpaw.core.matrix import Matrix
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.cgpaw import have_magma
from gpaw.gpu.diagonalization import CPUPYDiagonalizer, CuPyDiagonalizer, DiagonalizerOptions
from gpaw.gpu.diagonalization.magma_diagonalizer import MagmaDiagonalizer
from gpaw.test.gpu import assert_eigenpairs
from gpaw.mpi import world

if TYPE_CHECKING:
    from gpaw.gpu.diagonalization import GPUDiagonalizer
    from gpaw.mpi import MPIComm


# TODO: test with distributed matrices.
# The diagonalizers should still work but operate in serial

# Currently GPU distribution works only with blocksize = None, columns = 1

@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("matrix_size", [4])
@pytest.mark.parametrize("uplo", ['L', 'U'])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("diagonalizer_class",
                         [CPUPYDiagonalizer,
                          CuPyDiagonalizer,
                          MagmaDiagonalizer])
@pytest.mark.parametrize("distribution",
                         [None,
                          (world, -1, 1, None)])
def test_matrix_diagonalizer(fixt_raw_hermitian_matrix: cp.ndarray,
               diagonalizer_class: Type["GPUDiagonalizer"],
               matrix_size: int,
               # dist as in Matrix class: (comm, rows, cols, blocksize)
               distribution: Optional[tuple["MPIComm", int, int,
                                            Optional[int]]],
               dtype: np.dtype,
               uplo: str,
               inplace: bool):
    """"""

    if cupy_is_fake and diagonalizer_class is not CPUPYDiagonalizer:
        pytest.skip("CuPy is fake")
    elif not cupy_is_fake and diagonalizer_class is CPUPYDiagonalizer:
        pytest.skip("Not testing cpupy when running with real CuPy")

    if not have_magma and diagonalizer_class is MagmaDiagonalizer:
        pytest.skip("No MAGMA")

    raw_matrix = fixt_raw_hermitian_matrix(matrix_size, dtype=dtype,
                                           backend='cupy')
    non_distributed_matrix = Matrix(matrix_size, matrix_size, data=raw_matrix)

    #matrix = non_distributed_matrix.new(dist=distribution)
    matrix = Matrix(matrix_size, matrix_size, dtype=dtype, xp=cp, dist=distribution)

    non_distributed_matrix.redist(matrix)

    # try:
    #     non_distributed_matrix.redist(matrix)
    # except (AssertionError, ValueError):
    #     raise
    #     pytest.skip("Could not distribute matrix")

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
