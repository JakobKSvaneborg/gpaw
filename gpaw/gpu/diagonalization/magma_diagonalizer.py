from gpaw.gpu.diagonalization.diagonalizer import NonDistributedDiagonalizer, DiagonalizerOptions
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.new.timer import trace
from gpaw.utilities import as_real_dtype
from gpaw.cgpaw import have_magma

"""No logic in place yet for handling multi-GPU,
especially not distributed multi-GPU!
"""

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
        assert have_magma, "Must compile GPAW with MAGMA support"
        assert not cupy_is_fake, "Can't use MAGMA solvers with fake CuPy"

    @trace(gpu=True)
    def eigh_non_distributed(self,
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
        shape = inout_matrix.shape

        assert (inout_matrix.ndim == 2 and shape[0] == shape[1])

        # Alloc output arrays with CUPY.
        # Eigenvectors are real for symmetric/Hermitian matrices
        eigval_dtype = as_real_dtype(inout_matrix.dtype)
        eigvals = cp.empty((shape[0]), dtype=eigval_dtype)

        if options.inplace:
            eigvecs = inout_matrix
        else:
            eigvecs = cp.copy(inout_matrix)

        # This import only works if GPAW was compiled with MAGMA.
        # Doing the import here prevents crashes if importing this .py
        # module when MAGMA was not enabled during compilation.
        from gpaw.cgpaw import _eigh_magma_cupy

        _eigh_magma_cupy(eigvecs, eigvals, options.uplo)

        return eigvals, eigvecs
