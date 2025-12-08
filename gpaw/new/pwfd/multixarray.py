from gpaw.mpi import MPIComm
from gpaw.core.arrays import XArray
from typing import Sequence


class MultiXArray:
    """Object for working with wave-functions indexed by bands,
    k-points and spins as one vector.
    """
    def __init__(self,
                 a_unX: list[XArray],
                 comm: MPIComm,
                 weights: Sequence[float] | None = None):
        """K-points and spins are distributed over comm with weights.

        The weights default to ones.
        """
        self.a_unX = a_unX
        if weights is None:
            weights = [1.0] * len(a_unX)
        self.weights = weights
        self.comm = comm

    def new(self, a_unX: list[XArray]) -> MultiXArray:
        return MultiXArray(a_unX, self.comm, self.weights)

    def copy(self) -> MultiXArray:
        return self.new(
            [a_nX.copy() for a_nX in self.a_unX])

    def __neg__(self) -> MultiXArray:
        b_unX = self.copy()
        for b_nX in b_unX.a_unX:
            b_nX.data *= -1.0
        return b_unX

    def __iadd__(self, other: MultiXArray) -> MultiXArray:
        for a_nX, b_nX in zip(self.a_unX, other.a_unX):
            a_nX.data += b_nX.data
        return self

    def __sub__(self, other: MultiXArray) -> MultiXArray:
        a_unX = self.copy()
        for a_nX, b_nX in zip(a_unX.a_unX, other.a_unX):
            a_nX.data -= b_nX.data
        return a_unX

    def __mul__(self, other: float) -> MultiXArray:
        a_unX = self.copy()
        for a_nX in a_unX.a_unX:
            a_nX.data *= other
        return a_unX

    __rmul__ = __mul__

    def __matmul__(self, other: MultiXArray) -> float:
        return self.comm.sum_scalar(
            sum(weight * a_nX.trace_inner_product(b_nX)
                for weight, a_nX, b_nX
                in zip(self.weights, self.a_unX, other.a_unX)))

    def __setitem__(self, n, value: float):
        for a_nX in self.a_unX:
            a_nX.data[n] = value
