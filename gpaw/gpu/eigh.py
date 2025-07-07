from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake, is_hip
from gpaw.gpu.diagonalizer import (DiagonalizerOptions, GPUDiagonalizer,
                                   CPUPYDiagonalizer, CuPyDiagonalizer)
from gpaw.gpu.magma_diagonalizers import MagmaDiagonalizerSingleGPU
from gpaw.cgpaw import have_magma
from gpaw.new.timer import trace


@trace(gpu=True)
def gpu_eigh(inMatrix: cp.ndarray, UPLO: str) -> tuple[cp.ndarray, cp.ndarray]:
    """Solve eigensystem on GPUs.

    Usually CUDA > MAGMA > HIP, so we try to choose the best one.
    HIP native solver is questionably slow so for now do it on the CPU if
    MAGMA is not available.
    """

    diagonalizer: GPUDiagonalizer

    options = DiagonalizerOptions(uplo=UPLO)

    if cupy_is_fake:
        # Not actually using GPU. Return here already to reduce nesting
        diagonalizer = CPUPYDiagonalizer()
        eigvecs, eigvals = diagonalizer.eigh(inMatrix, options)
        return cp.asarray(eigvecs), cp.asarray(eigvals)

    #  Try to pick optimal backend
    use_magma = (is_hip and have_magma and inMatrix.shape[0] > 128
                 and inMatrix.ndim == 2)

    if use_magma:
        diagonalizer = MagmaDiagonalizerSingleGPU()

    else:
        diagonalizer = CuPyDiagonalizer()

    eigvecs, eigvals = diagonalizer.eigh(inMatrix, options)
    return cp.asarray(eigvecs), cp.asarray(eigvals)
