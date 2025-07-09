from gpaw.gpu.diagonalizer import GPUDiagonalizer, DiagonalizerOptions
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.new.timer import trace
from gpaw.utilities import as_real_dtype
from gpaw.cgpaw import have_magma

"""No logic in place yet for handling multi-GPU,
especially not distributed multi-GPU!
"""

class MagmaDiagonalizerSingleGPU(GPUDiagonalizer):
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
        assert have_magma, "Must compile GPAW with MAGMA support"
        #assert not cupy_is_fake, "Can't use MAGMA solvers with fake CuPy"

    @trace(gpu=True)
    def eigh(self,
             inout_matrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """
        Wrapper for MAGMA symmetric/Hermitian eigensolvers, GPU version.

        Parameters
        ----------
        inout_matrix : (N, N) cupy.ndarray
            The matrix to diagonalize. Must be symmetric or Hermitian.
            May be modified in-place depending on the `options` parameter.
        options : DiagonalizerOptions
            Options for the diagonalizer.

        Returns
        -------
        w : (N,) cupy.ndarray
            Eigenvalues in ascending order
        v : (N, N) cupy.ndarray
            Matrix containing orthonormal eigenvectors.
            Eigenvector corresponding to ``w[i]`` is in column ``v[:,i]``.
        """
        assert isinstance(inout_matrix, cp.ndarray)
        assert (inout_matrix.ndim == 2
                and inout_matrix.shape[0] == inout_matrix.shape[1]
        )

        ## PLAN: do distribution logic and input/output allocations on Python side

        # Alloc output arrays with CUPY.
        # Eigenvectors are real for symmetric/Hermitian matrices
        eigval_dtype = as_real_dtype(inout_matrix.dtype)
        eigvals = cp.empty((inout_matrix.shape[0]), dtype=eigval_dtype)

        if options.inplace:
            eigvecs = inout_matrix
        else:
            eigvecs = cp.copy(inout_matrix)

        # In-place eigensolver. Will throw if matrix dtype is unsupported
        from gpaw.cgpaw import _eigh_magma_gpu

        _eigh_magma_gpu(eigvecs, options.uplo, eigvals)

        return eigvals, eigvecs

        # # MAGMA eigenvectors are on rows, np/cp has them on columns.
        # # FIXME: how to avoid an intermediate copy when conj-transposing?
        # if options.inplace:
        #     _eigh_magma_gpu(inout_matrix, options.uplo, eigvals)
        #     inout_matrix = cp.conjugate(inout_matrix).T
        #     return eigvals, inout_matrix
        # else:
        #     eigvecs = cp.copy(inout_matrix)
        #     _eigh_magma_gpu(eigvecs, options.uplo, eigvals)
        #     return eigvals, cp.conjugate(eigvecs).T

