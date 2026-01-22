# Tight coupling with matrix.py... so need to be careful with circular imports
from typing import TYPE_CHECKING

from gpaw.cgpaw.gpu.magma import have_magma
from gpaw.gpu import cupy_is_fake, device_count, is_hip
from gpaw.gpu.diagonalization.diagonalizer import (CPUPYDiagonalizer,
                                                   CuPyDiagonalizer,
                                                   DiagonalizerOptions,
                                                   GPUDiagonalizer)
from gpaw.gpu.diagonalization.magma_diagonalizer import MagmaDiagonalizer

if TYPE_CHECKING:
    from gpaw.core.matrix import Matrix


def suggest_diagonalizer(matrix: "Matrix") -> tuple[GPUDiagonalizer,
                                                    DiagonalizerOptions]:
    """Attempts to choose a good GPU diagonalizer backend and options for the
    given matrix.
    """

    matrix_size = matrix.shape[0]
    options = DiagonalizerOptions()

    if cupy_is_fake:
        return CPUPYDiagonalizer(), options

    if not is_hip:
        # NVIDIA GPU. CuPy calls cuSolver which is typically very fast
        if matrix_size < 128:
            return CPUPYDiagonalizer(), options
        else:
            return CuPyDiagonalizer(), options

    # AMD GPU. Prefer CPU for small matrices, MAGMA for larger ones
    if matrix_size < 400:
        return CPUPYDiagonalizer(), options

    if have_magma:
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

    else:
        # No MAGMA on AMD, just use the CPU
        return CPUPYDiagonalizer(), options
