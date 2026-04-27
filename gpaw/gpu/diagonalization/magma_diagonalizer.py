import numpy as np
from typing import cast

from gpaw.cgpaw.gpu import magma
from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.gpu.diagonalization.diagonalizer import (DiagonalizerOptions,
                                                   NonDistributedDiagonalizer)
from gpaw.new.timer import trace
from gpaw.utilities import as_real_dtype


class MagmaDiagonalizer(NonDistributedDiagonalizer):
    """Single-GPU eigensolver using the MAGMA library. Note that this cannot
    be used if GPAW has not been compiled with MAGMA support.
    We check this in the constructor which fails with assert if MAGMA is not
    available.

    Importing the class is OK even without MAGMA, as long as no instances of
    it are created.
    """

    def __init__(self):
        """Constructor, asserts that both MAGMA and CuPy are available.
        This makes implementation details easier as we don't have to check
        for fake CuPy everywhere.
        """
        assert magma.is_available(), "Must compile GPAW with MAGMA support"
        assert not cupy_is_fake, "Can't use MAGMA solvers with fake CuPy"

    @trace(gpu=True)
    def eigh_non_distributed(self,
                             inout_matrix: cp.ndarray | np.ndarray,
                             options: DiagonalizerOptions
                             ) -> (tuple[cp.ndarray, cp.ndarray] |
                                   tuple[np.ndarray, np.ndarray]):
        """
        Wrapper for MAGMA symmetric/Hermitian eigensolvers.

        Parameters
        ----------
        inout_matrix : (N, N) Numpy or CuPy ndarray
            The matrix to diagonalize. Must be symmetric or Hermitian.
            May be modified in-place depending on the `options` parameter.
            Type (CuPy or Numpy) of this array determines type of the output
            arrays.
        options : DiagonalizerOptions
            Options for the diagonalizer.

        Returns
        -------
        w : (N,) ndarray
            Eigenvalues in ascending order.
        v : (N, N) ndarray
            Matrix containing orthonormal eigenvectors.
            Eigenvector corresponding to ``w[i]`` is in column ``v[:,i]``.
        """

        shape = inout_matrix.shape
        assert (inout_matrix.ndim == 2 and shape[0] == shape[1])

        xp = cp if isinstance(inout_matrix, cp.ndarray) else np

        # Eigenvectors are real for symmetric/Hermitian matrices
        eigval_dtype = as_real_dtype(inout_matrix.dtype)

        if options.inplace:
            eigvecs = inout_matrix
        else:
            eigvecs = xp.copy(inout_matrix)

        if options.gpus_per_process > 1 and xp is not np:
            # Handle multi-GPU with CuPy input. Magma needs the input on _host_
            host_matrix = cp.asnumpy(inout_matrix)
            eigvals = np.empty((shape[0]), dtype=eigval_dtype)

            magma.eigh_magma_numpy(
                host_matrix,
                eigvals,
                options.uplo,
                options.gpus_per_process)

            eigvecs[:] = cp.asarray(host_matrix)
            eigvals = cp.asarray(eigvals)

        else:
            eigvals = xp.empty((shape[0]), dtype=eigval_dtype)
            if xp is np:
                eigvecs = cast(np.ndarray, eigvecs)
                magma.eigh_magma_numpy(eigvecs,
                                       eigvals,
                                       options.uplo,
                                       options.gpus_per_process)
            else:
                eigvecs = cast(cp.ndarray, eigvecs)
                magma.eigh_magma_cupy(eigvecs, eigvals, options.uplo)

        # Transform to Numpy convention (conjugate transpose)
        xp.conjugate(eigvecs, out=eigvecs.T)

        return eigvals, eigvecs  # type: ignore
