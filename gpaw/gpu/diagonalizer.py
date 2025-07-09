from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.new.timer import trace

from warnings import warn
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DiagonalizerOptions:
    uplo: str = 'L'
    inplace: bool = False
    """If inplace is True, allows the diagonalizer to modify the input matrix
    in-place and replace it with the result eigenvectors.
    NB: CuPy diagonalizer ignores this option.
    """


class GPUDiagonalizer(ABC):
    """"""

    @abstractmethod
    def eigh(self,
             inout_matrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""
        pass


class CPUPYDiagonalizer(GPUDiagonalizer):
    """For cpupy"""

    def eigh(self,
             inout_matrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""

        if not cupy_is_fake:
            warn("Using CPUPYDiagonalizer with real CuPy -- why??")

        from scipy.linalg import eigh as scipy_eigh

        eigs, evals = scipy_eigh(cp.asnumpy(inout_matrix),
                                 lower=(options.uplo == 'L'),
                                 check_finite=False)

        return cp.asarray(eigs), cp.asarray(evals)


class CuPyDiagonalizer(GPUDiagonalizer):
    """"""

    @trace(gpu=True)
    def eigh(self,
             inout_matrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""

        # CuPy has no support for in-place eigh
        return cp.linalg.eigh(inout_matrix, options.uplo)
