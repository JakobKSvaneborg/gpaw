from gpaw.gpu.diagonalization.diagonalizer import (GPUDiagonalizer, DiagonalizerOptions,
                                   CPUPYDiagonalizer, CuPyDiagonalizer)
from gpaw.gpu.diagonalization.magma_diagonalizer import MagmaDiagonalizer
from gpaw.gpu import cupy_is_fake, is_hip, device_count
from gpaw.cgpaw import have_magma

# Tight coupling with matrix.py... so need to be careful with circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gpaw.core.matrix import Matrix


def suggest_diagonalizer(matrix: "Matrix")-> tuple[GPUDiagonalizer, DiagonalizerOptions]:
    """Tries to choose a good gpu diagonalizer backend and options for the
    given matrix.
    """

    if matrix.is_distributed():
        # Not implemented with GPUs, dunno what to do
        raise NotImplementedError("BLACS distribution not supported with GPUs," \
            "can't suggest a diagonalizer!")

    if cupy_is_fake:
        return CPUPYDiagonalizer(), DiagonalizerOptions()

    matrix_size = matrix.shape[0]
    if is_hip and have_magma and matrix_size > 128:
        # MAGMA should be faster than CuPy's version (which is rocSolver).
        # For large enough matrices it's also beneficial to use multiple GPUs

        options = DiagonalizerOptions()
        if device_count > 1:
            options.gpus_per_process = device_count

        return MagmaDiagonalizer(), options

    # None of the above: just use CuPy's method
    return CuPyDiagonalizer(), DiagonalizerOptions()
