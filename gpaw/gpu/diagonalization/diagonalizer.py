from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.new.timer import trace
from gpaw.utilities import as_real_dtype
from gpaw import debug

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from copy import copy
from typing import TYPE_CHECKING, Union
from warnings import warn

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
    """For multi-GPU."""


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

    def check_matrix(mat: Union[cp.ndarray, np.ndarray]):
        """"""

        xp = cp if isinstance(mat, cp.ndarray) else np

        # Check that the diagonal is real
        diagonal = xp.diag(mat)
        atol = 1e-6 if mat.dtype is np.complex64 else 1e-12

        if not xp.allclose(diagonal.imag, xp.zeros(diagonal.shape), atol=atol):
            warn("Using eigh() on matrix that has complex diagonal")

    @trace(gpu=True)
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

            """Problem: Can't gather/scatter if ratio N^2 / mpi_size is too
            large, because the 'count' parameter in MPI comms overflows!
            Temporary 'solution': Estimate it here and warn if a crash is to
            be expected.
            FIXME: use large-count versions of MPI comms (MPI-4)
            """
            nm = inout_matrix.dist.shape[0] * inout_matrix.dist.shape[1]

            # GPAW sends messages as MPI_BYTE
            msg_size = nm * inout_matrix.dtype.itemsize

            if msg_size > (2**31 - 1):
                warn("Matrix may be too large to gather over MPI! "
                     "If you crash here, try running with more MPI processes")

            matrix_non_distributed = inout_matrix.gather(0, broadcast=False)
            # Can always do in-place for the internal eigh
            options_non_distributed = copy(options)
            options_non_distributed.inplace = True
        else:
            matrix_non_distributed = inout_matrix
            options_non_distributed = options

        # avoid unnecessary copies for eigenvector output
        if needs_redist or options.inplace:
            eigvecs = matrix_non_distributed
        else:
            eigvecs = matrix_non_distributed.new()

        comm = matrix_non_distributed.dist.comm
        if comm.rank == 0:

            if debug:
                self.check_matrix(matrix_non_distributed.data)

            # NOTE: Very confusing that matrix.eigh wants to give the
            # eigenvector matrix as transposed, according to the old version.
            # So we do the same here... And ensure it remains C-contiguous

            eigvals, eigvecs.data.T[:] = (
                self.eigh_non_distributed(matrix_non_distributed.data,
                                          options_non_distributed)
            )

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
    """Diagonalizer that copies the matrix to CPU and calls scipy.linalg.eigh.
    """

    @trace
    def eigh_non_distributed(self,
                             inout_matrix: cp.ndarray,
                             options: DiagonalizerOptions
                             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""

        from scipy.linalg import eigh as scipy_eigh

        eigvals, eigvecs = scipy_eigh(cp.asnumpy(inout_matrix),
                                      lower=(options.uplo == 'L'),
                                      check_finite=False,
                                      overwrite_a=options.inplace)

        eigvals, eigvecs = cp.asarray(eigvals), cp.asarray(eigvecs)

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
