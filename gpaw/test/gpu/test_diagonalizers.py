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
from gpaw.new.c import GPU_AWARE_MPI
from gpaw.gpu.mpi import CuPyMPI

if TYPE_CHECKING:
    from gpaw.gpu.diagonalization import GPUDiagonalizer
    from gpaw.mpi import MPIComm

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
# Test both the "safe" CuPyMPI communicator and direct GPU-aware MPI (world)
@pytest.mark.parametrize("distribution",
                         [(CuPyMPI(world), -1, 1, None),
                          (world, -1, 1, None)])
def test_matrix_diagonalizer(fixt_raw_hermitian_matrix: cp.ndarray,
               diagonalizer_class: Type["GPUDiagonalizer"],
               matrix_size: int,
               # dist as in Matrix class: (comm, rows, cols, blocksize)
               distribution: tuple["MPIComm", int, int, Optional[int]],
               dtype: np.dtype,
               uplo: str,
               inplace: bool):
    """"""

    comm = distribution[0]

    if cupy_is_fake and diagonalizer_class is not CPUPYDiagonalizer:
        pytest.skip("CuPy is fake")
    elif not cupy_is_fake and diagonalizer_class is CPUPYDiagonalizer:
        pytest.skip("Not testing cpupy when running with real CuPy")

    if not GPU_AWARE_MPI and not isinstance(comm, CuPyMPI):
        pytest.skip("No GPU-aware MPI")

    if not have_magma and diagonalizer_class is MagmaDiagonalizer:
        pytest.skip("No MAGMA")

    # Matrix data to be wrapped in a distributed Matrix class
    raw_matrix: cp.ndarray = fixt_raw_hermitian_matrix(matrix_size,
                                                       dtype=dtype,
                                                       backend='cupy')

    matrix = Matrix.scatter(raw_matrix, distribution, 0)
    matrix_orig = matrix.copy()

    # Reference values
    eigvals_ref, eigvecs_ref = cp.linalg.eigh(raw_matrix, UPLO=uplo)

    diagonalizer = diagonalizer_class()
    options = DiagonalizerOptions(uplo=uplo, inplace=inplace)

    eigvals, eigvecs = diagonalizer.eigh(matrix, options)

    if options.inplace:
        assert eigvecs is matrix
    else:
        # Not in-place, matrix should stay unchanged
        cp.testing.assert_allclose(matrix_orig.data, matrix.data)

    # Gather eigenvectors back to root rank for checking
    eigvecs = eigvecs.gather(0, broadcast=False)

    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    cp.testing.assert_allclose(eigvals, eigvals_ref, atol=atol, rtol=rtol)

    if comm.rank == 0:
        # TODO: use Matrix.multiply and other direct Matrix routines.
        # The function below operates on raw xp.ndarrays, and we need to
        # transpose eigvecs back to original convention => very confusing
        assert_eigenpairs(raw_matrix, eigvals, eigvecs.data.T,
                          rtol=rtol, atol=atol)
