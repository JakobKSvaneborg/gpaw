from typing import TYPE_CHECKING

import numpy as np
import pytest

from gpaw.cgpaw.gpu.magma import have_magma
from gpaw.core.matrix import Matrix
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake, device_count
from gpaw.gpu.diagonalization import (CPUPYDiagonalizer, CuPyDiagonalizer,
                                      DiagonalizerOptions)
from gpaw.gpu.diagonalization.magma_diagonalizer import MagmaDiagonalizer
from gpaw.gpu.mpi import CuPyMPI
from gpaw.mpi import world
from gpaw.new.c import GPU_AWARE_MPI
from gpaw.test.gpu import assert_eigenpairs, fill_uplo

if TYPE_CHECKING:
    from gpaw.gpu.diagonalization import GPUDiagonalizer
    from gpaw.mpi import MPIComm


def diagonalizer_tester_common(
        raw_matrix: cp.ndarray,
        matrix: Matrix,
        diagonalizer_class: type["GPUDiagonalizer"],
        options: DiagonalizerOptions) -> None:
    """"""

    if cupy_is_fake and diagonalizer_class is not CPUPYDiagonalizer:
        pytest.skip("CuPy is fake")

    if not have_magma and diagonalizer_class is MagmaDiagonalizer:
        pytest.skip("No MAGMA")

    matrix_orig = matrix.copy()

    # Reference values
    eigvals_ref, eigvecs_ref = cp.linalg.eigh(raw_matrix, UPLO=options.uplo)

    diagonalizer = diagonalizer_class()
    eigvals, eigvecs = diagonalizer.eigh(matrix, options)

    if options.inplace:
        assert eigvecs is matrix
    else:
        # Not in-place, matrix should stay unchanged
        cp.testing.assert_allclose(matrix_orig.data, matrix.data)

    # Gather eigenvectors back to root rank for checking
    eigvecs = eigvecs.gather(0, broadcast=False)

    dtype = matrix.dtype
    atol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5
    rtol = 1e-12 if (dtype == np.float64 or dtype == np.complex128) else 1e-5

    cp.testing.assert_allclose(eigvals, eigvals_ref, atol=atol, rtol=rtol)

    # Test that the results really solve the intended eigenproblem
    true_matrix = fill_uplo(raw_matrix, options.uplo)

    if matrix.dist.comm.rank == 0:
        # TODO: use Matrix.multiply and other direct Matrix routines.
        # The function below operates on raw xp.ndarrays, and we need to
        # transpose eigvecs back to original convention => very confusing
        true_eigvecs = eigvecs.data.T
        assert_eigenpairs(true_matrix, eigvals, true_eigvecs,
                          rtol=rtol, atol=atol)

        # check orthonormality
        cp.testing.assert_allclose(cp.eye(matrix.shape[0]),
                                   true_eigvecs.T.conj() @ true_eigvecs,
                                   rtol=rtol, atol=atol)


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
def test_gpu_diagonalizer(fixt_eigh_test_matrix: cp.ndarray,
                          diagonalizer_class: type["GPUDiagonalizer"],
                          matrix_size: int,
                          # dist as in Matrix class: (comm, rows, cols, block)
                          distribution: tuple["MPIComm", int, int,
                                              int | None],
                          dtype: np.dtype,
                          uplo: str,
                          inplace: bool):
    """Test GPU eigensystem solvers."""

    comm = distribution[0]
    if not GPU_AWARE_MPI and not isinstance(comm, CuPyMPI):
        pytest.skip("No GPU-aware MPI")

    # Matrix data to be wrapped in a distributed Matrix class
    raw_matrix: cp.ndarray = fixt_eigh_test_matrix(matrix_size,
                                                   dtype=dtype,
                                                   backend='cupy')

    matrix = Matrix.scatter(raw_matrix, distribution, 0)

    options = DiagonalizerOptions(uplo=uplo, inplace=inplace)

    diagonalizer_tester_common(raw_matrix,
                               matrix,
                               diagonalizer_class,
                               options)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [np.float32, np.float64,
                                   np.complex64, np.complex128])
@pytest.mark.parametrize("matrix_size", [4, 156,])
@pytest.mark.parametrize("uplo", ['L', 'U'])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("diagonalizer_class",
                         [MagmaDiagonalizer])
@pytest.mark.parametrize("distribution",
                         [(CuPyMPI(world), -1, 1, None),
                          (world, -1, 1, None)])
def test_multigpu(fixt_eigh_test_matrix: cp.ndarray,
                  diagonalizer_class: type["GPUDiagonalizer"],
                  matrix_size: int,
                  # dist as in Matrix class: (comm, rows, cols, blocksize)
                  distribution: tuple["MPIComm", int, int, int | None],
                  dtype: np.dtype,
                  uplo: str,
                  inplace: bool):
    """Test multi-GPU diagonalization. Will attempt to utilize all available
    GPUs"""

    num_gpus = device_count

    if num_gpus < 2:
        pytest.skip("Not enough GPUs available")

    comm = distribution[0]
    if not GPU_AWARE_MPI and not isinstance(comm, CuPyMPI):
        pytest.skip("No GPU-aware MPI")

    raw_matrix: cp.ndarray = fixt_eigh_test_matrix(matrix_size,
                                                   dtype=dtype,
                                                   backend='cupy')

    matrix = Matrix.scatter(raw_matrix, distribution, 0)

    options = DiagonalizerOptions(uplo=uplo,
                                  inplace=inplace,
                                  gpus_per_process=num_gpus)

    diagonalizer_tester_common(raw_matrix,
                               matrix,
                               diagonalizer_class,
                               options)
