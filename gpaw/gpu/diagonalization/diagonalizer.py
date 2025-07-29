from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.new.timer import trace
from gpaw.utilities import as_real_dtype

from warnings import warn
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from gpaw.core.matrix import Matrix


@dataclass
class DiagonalizerOptions:
    """Collects configuration options to use with GPUDiagonalizer objects.
    """
    uplo: str = 'L'
    inplace: bool = False
    """If inplace is True, allows the diagonalizer to modify the input matrix
    in-place and replace it with the result eigenvectors. Use if memory usage
    is a concern. NB: Not all backends guarantee memory savings.
    """
    gpus_per_process: int = 1


class GPUDiagonalizer(ABC):
    """
    """

    @abstractmethod
    def eigh(self,
             inout_matrix: "Matrix",
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, "Matrix"]:
        """Eigensolver that is aware of matrix internal distribution.
        """
        pass


class NonDistributedDiagonalizer(GPUDiagonalizer):
    """Base diagonalizer class for solving eigensystems of non-distributed
    matrices. Everything is done on 1 CPU but possibly with multiple GPUs."""

    @abstractmethod
    def eigh_non_distributed(self,
                             inout_matrix: cp.ndarray,
                             options: DiagonalizerOptions
                             ) -> tuple[cp.ndarray, cp.ndarray]:
        """Solve eigenvalues and eigenvectors of a GPU matrix, represented by
        CuPy array.
        """
        pass

    def eigh(self,
             inout_matrix: "Matrix",
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, "Matrix"]:
        """"""

        assert isinstance(inout_matrix.data, cp.ndarray)
        assert inout_matrix.shape[0] == inout_matrix.shape[1]

        needs_redist = inout_matrix.is_distributed()

        if needs_redist:
            # We will do eigh on a non-distributed copy
            matrix_non_distributed = inout_matrix.gather(0, broadcast=False)
        else:
            matrix_non_distributed = inout_matrix

        # avoid unnecessary copies for eigenvector output
        if needs_redist or options.inplace:
            eigvecs = matrix_non_distributed
        else:
            eigvecs = matrix_non_distributed.new()

        comm = matrix_non_distributed.dist.comm
        if comm.rank == 0:
            eigvals, eigvecs.data[:] = (
                self.eigh_non_distributed(matrix_non_distributed.data, options)
            )

            # NOTE: Very confusing that matrix.eigh wants to give the
            # eigenvector matrix as transposed, according to the old version.
            # So we do the same here...
            eigvecs.data = eigvecs.data.T
        else:
            # Other ranks need to alloc recv buffers for eigenvalues (real!)
            eigvals = cp.empty(inout_matrix.shape[0],
                               dtype=as_real_dtype(inout_matrix.dtype))

        comm.broadcast(eigvals, 0)

        if not needs_redist:
            return eigvals, eigvecs

        # distribute back to the original layout
        if options.inplace:
            out_eigvecs = inout_matrix
        else:
            out_eigvecs = inout_matrix.new()

        eigvecs.redist(out_eigvecs)
        return eigvals, out_eigvecs


class CPUPYDiagonalizer(NonDistributedDiagonalizer):
    """For cpupy"""

    def eigh_non_distributed(self,
             inout_matrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""

        if not cupy_is_fake:
            warn("Using CPUPYDiagonalizer with real CuPy -- why??")

        from scipy.linalg import eigh as scipy_eigh

        eigvals, eigvecs = scipy_eigh(cp.asnumpy(inout_matrix),
                                        lower=(options.uplo == 'L'),
                                        check_finite=False)

        eigvals, eigvecs = cp.ndarray(eigvals), cp.ndarray(eigvecs)

        if options.inplace:
            inout_matrix = eigvecs

        return eigvals, eigvecs

class CuPyDiagonalizer(NonDistributedDiagonalizer):
    """"""

    def __init__(self):
        """Constructor, asserts that CuPy is available (not fake).
        """
        assert not cupy_is_fake, "Can't use CuPy diagonalizer with fake CuPy"


    @trace(gpu=True)
    def eigh_non_distributed(self,
             inout_matrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """CuPy does not support distributed matrices so this operates
        in serial. No memory savings with the 'inplace' option either.
        """

        eigvals, eigvecs = cp.linalg.eigh(inout_matrix, options.uplo)

        if options.inplace:
            # ensure that the input matrix and eigvecs are now the same object
            inout_matrix = eigvecs

        return eigvals, eigvecs
