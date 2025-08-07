from gpaw.gpu.diagonalization.diagonalizer import (GPUDiagonalizer,
                                                   DiagonalizerOptions,
                                                   CPUPYDiagonalizer,
                                                   CuPyDiagonalizer)
from gpaw.gpu.diagonalization.magma_diagonalizer import MagmaDiagonalizer
from gpaw.gpu import cupy_is_fake, is_hip, device_count
from gpaw.cgpaw import have_magma

# Tight coupling with matrix.py... so need to be careful with circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gpaw.core.matrix import Matrix


def suggest_diagonalizer(matrix: "Matrix") -> tuple[GPUDiagonalizer,
                                                    DiagonalizerOptions]:
    """Attempts to choose a good GPU diagonalizer backend and options for the
    given matrix.
    """

    matrix_size = matrix.shape[0]

    if cupy_is_fake or matrix_size < 400:
        return CPUPYDiagonalizer(), DiagonalizerOptions()

    if not is_hip:
        return CuPyDiagonalizer(), DiagonalizerOptions()

    if is_hip and have_magma:

        options = DiagonalizerOptions()
        if device_count > 1:
            # Multi-gpu can be faster for large matrices.
            # The following does some rudimentary GPU count selection.
            # TODO: improve this when we have more benchmarks

            # Tested for N <= 24k
            if matrix_size > 16000:
                options.gpus_per_process = min(device_count, 4)
            elif matrix_size > 10000:
                options.gpus_per_process = min(device_count, 3)
            elif matrix_size > 5000:
                options.gpus_per_process = min(device_count, 2)
            else:
                options.gpus_per_process = 1

        return MagmaDiagonalizer(), options

    # None of the above: just use CuPy's method
    return CuPyDiagonalizer(), DiagonalizerOptions()
