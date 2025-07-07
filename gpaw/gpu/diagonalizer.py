from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.new.timer import trace

from warnings import warn
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DiagonalizerOptions:
    uplo: str = 'L'


class GPUDiagonalizer(ABC):
    """"""

    @abstractmethod
    def eigh(self,
             inOutMatrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""
        pass


class CPUPYDiagonalizer(GPUDiagonalizer):
    """For cpupy"""

    def eigh(self,
             inOutMatrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""

        if not cupy_is_fake:
            warn("Using CPUPYDiagonalizer with real CuPy -- why??")

        from scipy.linalg import eigh as scipy_eigh

        eigs, evals = scipy_eigh(cp.asnumpy(inOutMatrix),
                                 lower=(options.uplo == 'L'),
                                 check_finite=False)

        return cp.asarray(eigs), cp.asarray(evals)


class CuPyDiagonalizer(GPUDiagonalizer):
    """"""

    @trace(gpu=True)
    def eigh(self,
             inOutMatrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""

        return cp.linalg.eigh(inOutMatrix, options.uplo)
