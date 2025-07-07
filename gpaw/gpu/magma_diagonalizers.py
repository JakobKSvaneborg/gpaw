from gpaw.gpu.diagonalizer import GPUDiagonalizer, DiagonalizerOptions
from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.new.timer import trace
from gpaw.new.magma import eigh_magma_gpu
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
        assert not cupy_is_fake, "Can't use MAGMA solvers with fake CuPy"

    @trace(gpu=True)
    def eigh(self,
             inOutMatrix: cp.ndarray,
             options: DiagonalizerOptions
             ) -> tuple[cp.ndarray, cp.ndarray]:
        """"""

        assert isinstance(inOutMatrix, cp.ndarray)

        ## TODO can get rid of the extra python layer, just call the C-function here
        return eigh_magma_gpu(inOutMatrix, options.uplo)
