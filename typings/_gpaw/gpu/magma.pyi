import cupy
import numpy
import typing

def eigh_magma_cupy(inout_matrix: cupy.ndarray, inout_eigenvalues: cupy.ndarray, uplo: str) -> None:
    """eigh_magma_cupy(inout_matrix: cupy.ndarray, inout_eigenvalues: cupy.ndarray, uplo: str) -> None


            Solves eigensystem on the GPU with Cupy input/output. This is an in-place solver:
            the input matrix will be overwritten with resulting eigenvectors. The input eigenvalue array
            must already be allocated to correct size (its contents don't matter).
            Input matrix is in Cupy/Scipy conventions. Output is still in MAGMA (Fortran) convention,
            so you will need to conjugate transpose to get back to Numpy conventions for eigenvectors.
    """
def eigh_magma_numpy(inout_matrix: numpy.ndarray, inout_eigenvalues: numpy.ndarray, uplo: str, num_gpus: typing.SupportsInt) -> None:
    """eigh_magma_numpy(inout_matrix: numpy.ndarray, inout_eigenvalues: numpy.ndarray, uplo: str, num_gpus: typing.SupportsInt) -> None


            Solves eigensystem on the GPU with Numpy input/output. This is an in-place solver:
            the input matrix will be overwritten with resulting eigenvectors. The input eigenvalue array
            must already be allocated to correct size (its contents don't matter).
            Input matrix is in Numpy/Scipy conventions. Output is still in MAGMA (Fortran) convention,
            so you will need to conjugate transpose to get back to Numpy conventions for eigenvectors.

            Passing num_gpus > 1 will instruct the solver to utilize multiple GPUs. This can be beneficial for large matrices (N >= 10k).
            Note though that the GPUs must be directly reachable from the same Cuda/HIP context, ie. this is a single-node solver without Scalapack-like support.
    """
def is_available() -> bool:
    """is_available() -> bool

    Returns true if MAGMA is available, false otherwise.
    """
def magma_finalize() -> None:
    """magma_finalize() -> None

    Cleanup of internal MAGMA state.
    """
def magma_init() -> None:
    """magma_init() -> None

    Initializes MAGMA library. Must be called come AFTER any calls to cudaSetValidDevices and cudaSetDeviceFlags.
            Call only if GPUs are available.
    """
