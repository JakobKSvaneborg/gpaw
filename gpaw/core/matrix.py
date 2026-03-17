"""BLACS distributed matrix object."""
from __future__ import annotations

from types import ModuleType

import numpy as np
import scipy.linalg as sla

import gpaw.cgpaw as cgpaw
import gpaw.utilities.blas as blas
from gpaw import debug
from gpaw.gpu import XP
from gpaw.gpu import cupy as cp
from gpaw.gpu import gpu_gemm
from gpaw.gpu.diagonalization import suggest_diagonalizer
from gpaw.mpi import MPIComm, _Communicator, serial_comm
from gpaw.typing import Array1D, Array2D, ArrayLike1D, ArrayLike2D

_global_blacs_context_store: dict[tuple[_Communicator, int, int], int] = {}


def suggest_blocking(N: int, ncpus: int) -> tuple[int, int, int]:
    """Suggest blocking of ``NxN`` matrix.

    Returns rows, columns, blocksize tuple.

    >>> suggest_blocking(10, 6)
    (3, 2, 2)
    """

    if ncpus == 1:
        return 1, 1, 0

    nprow = ncpus
    npcol = 1

    # Make npcol and nprow as close to each other as possible
    npcol_try = npcol
    while npcol_try < nprow:
        if ncpus % npcol_try == 0:
            npcol = npcol_try
            nprow = ncpus // npcol
        npcol_try += 1

    assert npcol * nprow == ncpus

    # ScaLAPACK creates trouble if there aren't at least a few whole blocks.
    # Choose block size so that there will always be at least one whole block
    # and at least two blocks in total.
    blocksize = max((N - 2) // max(nprow, npcol), 1)
    # The next commented line would give more whole blocks.
    # blocksize = max(N // max(nprow, npcol) - 2, 1)

    # Use block size that is a power of 2 and at most 64
    blocksize = 2**int(np.log2(blocksize))
    blocksize = max(min(blocksize, 64), 1)

    return nprow, npcol, blocksize


class MatrixWithNoData:
    def __init__(self,
                 M: int,
                 N: int,
                 *,
                 dtype=None,
                 dist: MatrixDistribution | tuple | None = None,
                 xp=np):
        self.shape = (M, N)
        self.dtype = dtype
        self.data = np.empty((0, 0), dtype)
        dist = dist or ()
        if isinstance(dist, tuple):
            kwargs = {key: val for key, val in zip(['comm', 'r', 'c', 'b'],
                                                   dist)}
            dist = create_distribution(M, N, **kwargs)
        self.dist = dist
        self.xp = xp

    def create(self) -> Matrix:
        return Matrix(
            *self.shape, dtype=self.dtype, dist=self.dist, xp=self.xp)


class Matrix(XP):
    def __init__(self,
                 M: int,
                 N: int,
                 *,
                 dtype=None,
                 data: ArrayLike2D | None = None,
                 dist: MatrixDistribution | MPIComm | tuple | None = None,
                 xp=None):
        """Matrix object.

        Parameters
        ----------
        M:
            Rows.
        N:
            Columns.
        dtype:
            Data type (float or complex).
        dist:
            BLACS distribution given as
            (communicator, rows, columns, blocksize)
            tuple.  Default is None meaning no distribution.
        data:
            Numpy ndarray to use for storage.  By default, a new ndarray
            will be allocated.
            """
        self.shape = (M, N)

        if data is None or isinstance(data, (np.ndarray, cp.ndarray)):
            pass
        else:
            data = np.asarray(data)

        if dtype is None:
            if data is None:
                dtype = float
            else:
                dtype = data.dtype
        self.dtype = np.dtype(dtype)
        assert np.dtype(self.dtype) in \
            [np.float32, np.float64, np.complex64, np.complex128], dtype

        self.xp: ModuleType
        if xp is None:
            if data is not None:
                xp = np if isinstance(data, np.ndarray) else cp
            elif isinstance(dist, CuPyDistribution):
                xp = cp
            else:
                xp = np
        super().__init__(xp)

        if isinstance(dist, MatrixDistribution):
            assert self.shape == dist.full_shape
            dist = dist.to_xp(xp)  # make sure xp and dist match
        else:
            if dist is None:
                dist = ()
            elif not isinstance(dist, tuple):
                dist = (dist,)
            kwargs = {
                key: val for key, val in zip(['comm', 'r', 'c', 'br', 'bc'],
                                             dist)}
            dist = create_distribution(M, N, xp=self.xp, **kwargs)
        self.dist = dist

        self.data: Array2D
        if data is None:
            self.data = self.xp.empty(dist.shape, self.dtype)
        else:
            assert data.shape == dist.shape, (data.shape, dist.shape)
            self.data = data

    def __repr__(self):
        dist = str(self.dist).split('(')[1]
        if self.xp is cp:
            dist = 'xp=cp, ' + dist
        return f'Matrix({self.dtype.name}: {dist}'

    def new(self, dist='inherit', data=None) -> Matrix:
        """Create new matrix of same shape and dtype.

        Default is to use same BLACS distribution.  Use dist to use another
        distribution.
        """
        return Matrix(*self.shape,
                      dtype=self.dtype,
                      dist=self.dist if dist == 'inherit' else dist,
                      data=data,
                      xp=self.xp)

    def copy(self) -> Matrix:
        """Create a copy."""
        M = self.new()
        M.data[:] = self.data
        return M

    def __setitem__(self, item, value):
        assert item == slice(None)
        assert isinstance(value, Matrix)
        self.data[:] = value.data

    def __iadd__(self, other):
        if isinstance(other, Matrix):
            other = other.data
        self.data += other
        return self

    def is_distributed(self) -> bool:
        """True if this matrix has nontrivial BLACS or GPU distribution.
        """
        return self.dist.shape != self.shape

    def multiply(self,
                 other,
                 alpha=1.0,
                 opa='N',
                 opb='N',
                 out=None,
                 data_buffer=None,
                 beta=0.0,
                 symmetric=False) -> Matrix:
        """BLAS matrix-multiplication with other matrix."""
        if not isinstance(other, Matrix):
            other = other.matrix
        A = self
        B = other
        dist = self.dist
        if out is None:
            assert beta == 0.0
            M = A.shape[0] if opa == 'N' else A.shape[1]
            N = B.shape[1] if opb == 'N' else B.shape[0]
            out = Matrix(M, N,
                         dtype=A.dtype,
                         xp=self.xp,
                         dist=dist.comm)
        elif not isinstance(out, Matrix):
            out = out.matrix
        if out.data is other.data:
            # Repeatably call multiply using data_buffer
            assert opa == 'N', 'Not implemented'
            assert opb == 'N', 'Not implemented'
            assert not beta, 'Not implemented'
            assert other.shape[0] == self.shape[0]

            # Assert simple (only row distributed) distributions:
            assert self.shape[1] == self.data.shape[1]
            assert other.shape[1] == other.data.shape[1]
            assert out.shape[1] == out.data.shape[1]

            if data_buffer is None:
                raise ValueError('other is out, and data_buffer is None')

            assert isinstance(data_buffer, other.xp.ndarray)
            dtype = other.data.dtype
            data_buffer = data_buffer.view(dtype)
            if other.data.shape[0] > 0:
                # Obtain buffer size s.t. the maximum number of
                # columns in other.data fits into data_buffer
                buffer_size = min(
                    data_buffer.size // other.data.shape[0],
                    other.data.shape[1])
            else:
                # There is no data in other. Thus buffer_size
                # fits all.
                buffer_size = other.data.shape[1]
            buffer_size = dist.comm.min_scalar(buffer_size)
            max_B = other.data.shape[1]

            if buffer_size >= max_B:
                # No need for sliced multiply
                other_buffer = other.new(
                    data=data_buffer[:other.data.size].reshape(
                        other.data.shape))
                other_buffer.data[:] = other.data
                dist.multiply(alpha, A, opa, other_buffer, opb, beta, out,
                              symmetric=symmetric)
                return out

            # Sliced multiply
            for i in range(0, max_B, buffer_size):
                r_buffer_size = min(max(other.data.shape[1] - i, 0),
                                    buffer_size)
                l_buffer_size = r_buffer_size * other.data.shape[0]
                buffer = Matrix(
                    M=other.shape[0],
                    N=r_buffer_size,
                    data=data_buffer[
                        :l_buffer_size].reshape(
                        (other.data.shape[0], r_buffer_size)),
                    dist=dist.comm,  # .new(M=other.shape[0], N=r_buffer_size),
                    xp=other.xp)
                buffer.data[:] \
                    = other.data[:, i:i + buffer_size]
                out_view = buffer.new(
                    data=out.data[:, i:i + buffer_size])
                dist.multiply(alpha, A, opa, buffer,
                              opb, beta, out_view, symmetric=False)
            return out

        dist.multiply(alpha, A, opa, B, opb, beta, out, symmetric=symmetric)
        return out

    def redist(self, other: Matrix) -> None:
        """Redistribute to other BLACS layout.
        `other` is the output, newly distributed matrix."""
        if self is other:
            return

        d1 = self.dist
        d2 = other.dist
        n1 = d1.rows * d1.columns
        n2 = d2.rows * d2.columns

        if n1 == n2 == 1:
            other.data[:] = self.data
            return

        if self.xp is np:
            c = d1.comm if d1.comm.size > d2.comm.size else d2.comm
            n = max(n1, n2)
            M, N = self.shape
            d1 = create_distribution(M, N, c,
                                     d1.rows, d1.columns, d1.br, d1.bc)
            d2 = create_distribution(M, N, c,
                                     d2.rows, d2.columns, d2.br, d2.bc)
            if n1 == n:
                ctx = d1.desc[1]
            else:
                ctx = d2.desc[1]
            redist(d1, self.data, d2, other.data, ctx)
            return

        if d2.all_data_on_rank_zero and d1.simple:
            comm = d1.comm
            if comm.rank == 0:
                M = self.shape[0]
                m = (M + d1.rows - 1) // d1.rows
                other.data[:m] = self.data
                for r in range(1, d1.rows):
                    m1 = min(r * m, M)
                    m2 = min(m1 + m, M)
                    comm.receive(other.data[m1:m2], r * d1.columns)
            elif comm.rank % d1.columns == 0:
                comm.send(self.data, 0)
            return

        if d1.all_data_on_rank_zero and d2.simple:
            comm = d2.comm
            if comm.rank == 0:
                M = self.shape[0]
                m = (M + d2.rows - 1) // d2.rows
                other.data[:] = self.data[:m]
                for r in range(1, d2.rows):
                    m1 = min(r * m, M)
                    m2 = min(m1 + m, M)
                    comm.send(self.data[m1:m2], r * d2.columns)
            elif comm.rank % d2.columns == 0:
                comm.receive(other.data, 0)
            return
        1 / 0

    def gather(self, root: int = 0, *, broadcast=False) -> Matrix:
        """Gather the Matrix on the root rank.

        Returns a new Matrix distributed so that all data is on the root rank
        """
        assert root == 0

        if self.dist.all_data_on_rank_zero:
            return self

        S = self.new(dist=(self.dist.comm, 1, 1, *self.shape))
        self.redist(S)
        if broadcast:
            if self.dist.comm.rank != 0:
                S = self.new(dist=None)
            self.dist.comm.broadcast(S.data, 0)
        return S

    @staticmethod
    def scatter(data: Array2D,
                *,
                dist: tuple[_Communicator, int, int, int | None],
                root: int = 0) -> Matrix:
        """Construct a distributed Matrix object by scattering a raw 2D array
        from 'root' rank. The 'dist' argument must specify the communicator
        and wanted distribution in same way as in the Matrix constructor
        Empty 'dist' argument is not allowed!
        """
        assert root == 0
        rows, cols = data.shape
        xp = cp if isinstance(data, cp.ndarray) else np
        matrix = Matrix(rows, cols, dtype=data.dtype, xp=xp, dist=dist)
        # Some acrobatics needed to bypass limitations in Matrix.redist()

        non_distributed_matrix = Matrix(rows, cols, data=data)
        non_distributed_matrix.redist(matrix)
        return matrix

    def inv(self, uplo='L') -> None:
        """Inplace inversion."""
        assert uplo == 'L'
        M, N = self.shape
        assert M == N
        dist = self.dist
        if dist.comm.size == 1:
            self.tril2full()
            self.data[:] = sla.inv(self.data,
                                   overwrite_a=True,
                                   check_finite=debug)
            return
        bc, br = dist.desc[4:6]
        assert bc == br
        info = cgpaw.scalapack_inverse(self.data, dist.desc, 'U')
        if info != 0:
            raise ValueError(f'scalapack_inverse error: {info}')

    def invcholesky(self) -> None:
        """In-place inverse of Cholesky decomposition.

        Calculate a lower triangle matrix `L` where:::

             †
          LSL = 1,

        and `S` is self.  Only the lower part of `S` is used.

        >>> S = Matrix(2, 2, data=[[1.0, np.nan],
        ...                        [0.1, 1.0]])
        >>> S.invcholesky()
        >>> S.data
        array([[ 1.        ,  0.        ],
               [-0.10050378,  1.00503782]])
        """
        S = self.gather()
        if self.dist.comm.rank == 0:
            if isinstance(S.data, np.ndarray):
                if debug:
                    S.data[np.triu_indices(S.shape[0], 1)] = 42.0
                L_nn = sla.cholesky(S.data,
                                    lower=True,
                                    overwrite_a=True,
                                    check_finite=debug)
                S.data[:] = sla.inv(L_nn,
                                    overwrite_a=True,
                                    check_finite=debug)
            else:
                S.tril2full()
                L_nn = cp.linalg.cholesky(S.data)
                S.data[:] = cp.linalg.inv(L_nn)

        if S is not self:
            S.redist(self)

    def eigh(self,
             S: Matrix | None = None,
             *,
             cc: bool = False,
             scalapack: tuple[MPIComm, int, int, int] = (serial_comm, 1, 1, 0),
             limit: int | None = None) -> Array1D:
        """Calculate eigenvectors and eigenvalues.

        Matrix must be symmetric/hermitian and stored in lower half.
        If ``S`` is given, solve a generalized eigenvalue problem.

        Parameters
        ----------
        cc: bool
            Complex conjugate matrix before finding eigenvalues.
        scalapack: tuple
            BLACS distribution for ScaLapack to use.  Default is to do serial
            diagonalization.
        limit:
            Number of eigenvector and values to find.  Defaults to all.
        """

        if limit == self.shape[0]:
            limit = None

        if limit:
            eigs = self.xp.empty(limit)
        else:
            eigs = self.xp.empty(self.shape[0])

        slcomm, rows, columns, blocksize = scalapack

        if rows * columns == 1:
            if self.dist.all_data_on_rank_zero:
                H = self
            else:
                H = self.gather()
                if S is not None:
                    S = S.gather()
            if self.dist.comm.rank == 0:
                self.dist.eigh_serial(H, S, eigs, cc, limit)
            self.dist.comm.broadcast(eigs, 0)
            if not self.dist.all_data_on_rank_zero:
                H.redist(self)
            return eigs

        H = self.new(dist=scalapack)
        self.redist(H)
        if S is not None:
            S0 = S
            S = H.new()
            S0.redist(S)

        if self.dist.comm.rank < rows * columns:
            eigh_parallel(H, S, eigs, cc, limit)

        # necessary to broadcast eps when some ranks are not used
        # in current scalapack parameter set
        # eg. (2, 1, 2) with 4 processes
        if 1:  # rows * columns < self.dist.comm.size:
            # looks like scalapack_general_diagonalize_dc needs this
            # always!  Not sure about scalapack_diagonalize_dc?
            self.dist.comm.broadcast(eigs, 0)

        H.redist(self)
        return eigs

    def eighl(self,
              L: Matrix,
              comm2: MPIComm = serial_comm) -> Array1D:
        """Solve generalized eigenvalue problem.

        With `H` being self, we solve for the eigenvectors `C` and the
        eigenvalues `Λ` (a diagonal matrix):::

          HC = SCΛ,

        where `L` is a lower triangle matrix such that:::

             †
          LSL = 1.

        The solution has these three steps:::

           ~      †   ~~   ~         †~
           H = LHL ,  HC = CΛ,  C = L C.

        Note that `H` must be the full matrix not just half of it!

        """
        M, N = self.shape
        assert M == N
        comm = self.dist.comm
        H = self

        if comm2.rank == 0:
            LH = L.multiply(H)
            LH.multiply(L, opb='C', out=H)
            r, c, b = suggest_blocking(M, comm.size)
            eig_n = H.eigh(scalapack=(comm, r, c, b))
            L.multiply(H, opa='C', opb='T', out=LH)
            H.data[:] = LH.data
        else:
            eig_n = np.empty(M)

        comm2.broadcast(eig_n, 0)
        comm2.broadcast(H.data, 0)
        return eig_n

    def complex_conjugate(self) -> None:
        """Inplace complex conjugation."""
        if np.issubdtype(self.dtype, np.complexfloating):
            self.xp.negative(self.data.imag, self.data.imag)

    def add_hermitian_conjugate(self, scale: float = 1.0) -> None:
        """Add hermitian conjugate to myself."""
        if self.dist.comm.size == 1:
            if scale != 1.0:
                self.data *= scale
            self.data += self.data.conj().T
            return
        self.dist.add_hermitian_conjugate(self, scale)

    def tril2full(self) -> None:
        """Fill in upper triangle from lower triangle.

        For a real matrix::

          a ? ?    a b d
          b c ? -> b c e
          d e f    d e f

        For a complex matrix, the complex conjugate of the lower part will
        be inserted into the upper part.
        """
        M, N = self.shape
        assert M == N

        dist = self.dist

        if dist.all_data_on_rank_zero:
            if dist.comm.rank == 0:
                lower = self.xp.tri(M, k=-1, dtype=bool)
                self.data.T[lower] = self.data[lower].conj()
            return

        desc = dist.desc
        cgpaw.scalapack_set(self.data, desc, 0.0, 0.0, 'L', M - 1, M - 1, 2, 1)
        buf = self.data.copy()
        # Set diagonal to zero in the copy:
        cgpaw.scalapack_set(buf, desc, 0.0, 0.0, 'L', M, M, 1, 1)
        # Now transpose tmp_mm adding the result to the original matrix:
        cgpaw.pblas_tran(M, M, 1.0, buf, 1.0, self.data, desc, desc, True)

    def add_to_diagonal(self, d: ArrayLike1D | float) -> None:
        """Add list of numbers or single number to diagonal of matrix."""
        n1, n2 = self.dist.my_row_range()
        M, N = self.shape
        assert M == N
        self.data.ravel()[n1::N + 1] += d

    def to_cpu(self) -> Matrix:
        """Create new matrix object with values transferred from GPU to CPU."""
        return self.to_xp(np)

    def to_xp(self, xp) -> Matrix:
        """Create new matrix object with data on GPU or CPU."""
        if xp is self.xp:
            assert xp is np, 'cp -> cp should not be needed!'
            return self
        if xp is np:
            return self.dist.matrix(data=cp.asnumpy(self.data))
        return self.dist.matrix(data=cp.asarray(self.data))

    def to_dtype(self, dtype) -> Matrix:
        """Convert to new data type."""
        if dtype == self.dtype:
            return self
        return self.dist.matrix(data=self.data.astype(dtype))


def _matrix(M):
    """Dig out Matrix object from wrapper(s)."""
    if isinstance(M, Matrix):
        return M
    return _matrix(M.matrix)


def redist(dist1, M1, dist2, M2, context):
    cgpaw.scalapack_redist(dist1.desc, dist2.desc,
                           M1, M2,
                           dist1.desc[2], dist1.desc[3],
                           1, 1, 1, 1,  # 1-indexing
                           context, 'G')


def create_distribution(M: int,
                        N: int,
                        comm: MPIComm | None = None,
                        r: int = -1,
                        c: int = 1,
                        br: int = 0,
                        bc: int = 0,
                        xp=np) -> MatrixDistribution:
    assert not (r == -1 and c == -1)
    assert r == -1 or r > 0
    assert c == -1 or c > 0

    comm = comm or serial_comm

    if r == -1:
        r = comm.size // c
    elif c == -1:
        c = comm.size // r

    if br == 0 and bc == 0:
        br = max(1, (M + r - 1) // r)
        bc = max(1, (N + c - 1) // c)
    elif bc == 0:
        bc = br

    if xp is cp:
        comm = comm or serial_comm
        return CuPyDistribution(M, N, comm, r, c, br, bc)

    if comm.size == 1:
        return NoDistribution(M, N)

    return BLACSDistribution(M, N, comm, r, c, br, bc)


class MatrixDistribution:
    comm: MPIComm
    rows: int
    columns: int
    br: int
    bc: int
    shape: tuple[int, int]
    full_shape: tuple[int, int]
    desc: Array1D
    simple = True
    all_data_on_rank_zero = True

    def matrix(self, dtype=None, data=None):
        return Matrix(*self.full_shape, dtype=dtype, data=data, dist=self)

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        raise NotImplementedError

    def eighl(self, H, L):
        raise NotImplementedError

    def new(self, M, N):
        raise NotImplementedError

    def to_xp(self, xp) -> MatrixDistribution:
        raise NotImplementedError

    def my_row_range(self) -> tuple[int, int]:
        """Return indices for range of my rows.

        >>> Matrix(2, 2).dist.my_row_range()
        (0, 2)
        """
        M, N = self.full_shape
        b = (M + self.rows - 1) // self.rows
        ok = (self.rows == self.comm.size and
              self.columns == 1 and
              self.br == b and
              self.bc == N)
        if not ok:
            raise ValueError(f'Can not create slice of distribution: {self}')
        n1 = min(self.comm.rank * b, M)
        n2 = min(n1 + b, M)
        return n1, n2

    def add_hermitian_conjugate(self,
                                a: Matrix,
                                scale: float) -> None:
        raise NotImplementedError

    def eigh_serial(self,
                    H: Matrix,
                    S: Matrix | None,
                    eigs: np.ndarray,
                    cc: bool = False,
                    limit: int | None = None) -> None:
        if cc and np.issubdtype(H.dtype, np.complexfloating):
            np.negative(H.data.imag, H.data.imag)
        if debug:
            H.data[np.triu_indices(H.shape[0], 1)] = 42.0
        if S is None:
            eigs[:], H.data.T[:] = sla.eigh(
                H.data,
                lower=True,
                overwrite_a=True,
                check_finite=debug,
                driver='evx' if H.data.size == 1 else 'evd')
        else:
            if debug:
                S.data[np.triu_indices(H.shape[0], 1)] = 42.0
            eigs[:], evecs = sla.eigh(
                H.data,
                S.data,
                lower=True,
                overwrite_a=True,
                overwrite_b=True,
                check_finite=debug,
                subset_by_index=(0, limit - 1) if limit else None)
            limit = limit or len(eigs)
            H.data.T[:, :limit] = evecs


class NoDistribution(MatrixDistribution):
    comm = serial_comm
    rows = 1
    columns = 1

    def __init__(self, M: int, N: int):
        self.shape = (M, N)
        self.full_shape = (M, N)
        self.br = M
        self.bc = N

    def __str__(self):
        return 'NoDistribution({}x{})'.format(*self.shape)

    def to_xp(self, xp) -> MatrixDistribution:
        if xp is np:
            return self
        return CuPyDistribution(*self.shape, serial_comm, 1, 1, *self.shape)

    def global_index(self, n):
        return n

    def new(self, M, N):
        return NoDistribution(M, N)

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        if symmetric:
            if opa == 'N':
                assert opb == 'C' or opb == 'T' and a.dtype == float
                if a is b:
                    blas.rk(alpha, a.data, beta, c.data)
                else:
                    if beta == 1.0 and a.shape[1] == 0:
                        return
                    blas.r2k(0.5 * alpha, a.data, b.data, beta, c.data)
            else:
                1 / 0
                assert opa == 'C' and opb == 'N'
                assert a is not b
                blas.r2k(0.5 * alpha, a.data, b.data, beta, c.data, 'n')

        else:
            blas.mmm(alpha, a.data, opa, b.data, opb, beta, c.data)


class BLACSDistribution(MatrixDistribution):
    serial = False

    def __init__(self, M, N, comm, r, c, br, bc):
        self.comm = comm
        self.rows = r
        self.columns = c
        self.full_shape = (M, N)
        self.br = br
        self.bc = bc

        key = (comm, r, c)
        context = _global_blacs_context_store.get(key)
        if context is None:
            try:
                context = cgpaw.new_blacs_context(comm.get_c_object(),
                                                  c, r, 'C')
            except AttributeError:
                pass
            else:
                _global_blacs_context_store[key] = context

        self.simple = (c == 1 and
                       br == (M + r - 1) // r and
                       bc == N)
        self.all_data_on_rank_zero = (br == M and bc == N)

        if context is None:
            assert c == 1
            assert br == (M + r - 1) // r
            n = N
            m = min((comm.rank + 1) * br, M) - min(comm.rank * br, M)
        else:
            n, m = cgpaw.get_blacs_local_shape(context, N, M, bc, br, 0, 0)
        if n < 0 or m < 0:
            n = m = 0
        self.shape = (m, n)
        lld = max(1, n)
        if context is not None:
            self.desc = np.array([1, context, N, M, bc, br, 0, 0, lld],
                                 np.intc)

    def __str__(self):
        return ('BLACSDistribution(global={}, local={}, blocksize={})'
                .format(*('{}x{}'.format(*shape)
                          for shape in [self.desc[3:1:-1],
                                        self.shape,
                                        self.desc[5:3:-1]])))

    def global_index(self, myi):
        return self.comm.rank * int(self.desc[5]) + myi

    def new(self, M, N):
        return BLACSDistribution(M, N,
                                 self.comm,
                                 self.rows, self.columns,
                                 self.br, self.bc)

    def multiply(self, alpha, a, opa, b, opb, beta, c, symmetric):
        if self.comm.size > 1:
            ok = a.dist.simple and b.dist.simple and c.dist.simple
            if ok:
                # Special cases that don't need scalapack - most likely also
                # faster:
                if opa == 'N' and opb == 'N':
                    return mmm_nn(a, b, c, alpha, beta, blas.mmm)
                if opa == 'N' and opb == 'C':
                    if symmetric:
                        if beta == 1.0:
                            return mmm_nc_sym(a, b, c, alpha, blas.mmm)
                    else:
                        return mmm_nc(a, b, c, alpha, beta, blas.mmm)

        if symmetric:
            assert opa == 'N'
            assert opb == 'C' or opb == 'T' and a.dtype == float
            N, K = a.shape
            if a is b:
                cgpaw.pblas_rk(N, K, alpha, a.data,
                               beta, c.data,
                               a.dist.desc, c.dist.desc,
                               'U')
            else:
                cgpaw.pblas_r2k(N, K, 0.5 * alpha, b.data, a.data,
                                beta, c.data,
                                b.dist.desc, a.dist.desc, c.dist.desc,
                                'U')
        else:
            Ka, M = a.shape
            N, Kb = b.shape
            if opa == 'N':
                Ka, M = M, Ka
            if opb == 'N':
                N, Kb = Kb, N
            cgpaw.pblas_gemm(N, M, Ka, alpha, b.data, a.data,
                             beta, c.data,
                             b.dist.desc, a.dist.desc, c.dist.desc,
                             opb, opa)
        return c

    def add_hermitian_conjugate(self,
                                a: Matrix,
                                scale: float) -> None:
        tmp = a.copy()
        cgpaw.pblas_tran(*self.full_shape, scale, tmp.data, scale, a.data,
                         self.desc, self.desc, True)

    def to_xp(self, xp) -> MatrixDistribution:
        if xp is np:
            return self
        return CuPyDistribution(
            *self.full_shape,
            self.comm, self.rows, self.columns, self.br, self.bc)


def eigh_parallel(H: Matrix,
                  S: Matrix | None,
                  eigs: np.ndarray,
                  cc: bool = False,
                  limit: int | None = None) -> None:
    array = H.data.copy()
    if not cc and np.issubdtype(H.dtype, np.complexfloating):
        np.negative(array.imag, array.imag)
    eigs0 = np.empty(H.shape[0]) if limit else eigs
    if S is None:
        info = cgpaw.scalapack_diagonalize_dc(
            array, H.dist.desc, 'U', H.data, eigs0)
    else:
        sarray = S.data
        if not cc and np.issubdtype(H.dtype, np.complexfloating):
            np.negative(sarray.imag, sarray.imag)
        info = cgpaw.scalapack_general_diagonalize_dc(
            array, H.dist.desc, 'U', sarray, H.data, eigs0)
    if limit:
        eigs[:] = eigs0[:limit]
    assert info == 0, info


def cublas_mmm(alpha, a, opa, b, opb, beta, c):
    if c.size == 0:
        return
    if a.size == 0 and beta == 1.0:
        return
    gpu_gemm(opa.replace('C', 'H'), opb.replace('C', 'H'),
             a, b, c, alpha, beta)


class CuPyDistribution(MatrixDistribution):
    def __init__(self, M, N, comm, r, c, br, bc):
        self.comm = comm
        self.rows = r
        self.columns = c
        self.full_shape = (M, N)
        self.br = br
        self.bc = bc
        assert bc == max(1, N)
        row = comm.rank // c
        if br >= M:
            m = M if row == 0 else 0
        elif br == (M + r - 1) // r:
            m = min((row + 1) * br, M) - min(row * br, M)
            self.all_data_on_rank_zero = False
        else:
            raise ValueError
        self.shape = (m, N)

    def __str__(self):
        M, N = self.full_shape
        m, N = self.shape
        return f'CuPyDistribution(global={M}x{N}, local={m}x{N})'

    def to_xp(self, xp):
        if xp is not np:
            return self
        if self.comm.size == 1:
            return NoDistribution(*self.full_shape)
        return BLACSDistribution(
            *self.full_shape,
            self.comm, self.rows, self.columns, self.br, self.bc)

    def global_index(self, n):
        1 / 0
        return n

    def new(self, M, N):
        return CuPyDistribution(M, N,
                                self.comm,
                                self.rows, self.columns, self.br, self.bc)

    def multiply(self, alpha, a, opa, b, opb, beta, c, *, symmetric=False):
        if self.comm.size > 1:
            if opa == 'N' and opb == 'N':
                return mmm_nn(a, b, c, alpha, beta, cublas_mmm)
            if opa == 'N' and opb == 'C':
                if symmetric:
                    if beta == 1.0:
                        return mmm_nc_sym(a, b, c, alpha, cublas_mmm)
                else:
                    return mmm_nc(a, b, c, alpha, beta, cublas_mmm)
            if opa == 'C' and opb == 'T' and beta == 0.0:
                # Quick'n'dirty hack:
                a = a.gather()
                b = b.gather()
                c0 = b.new()
                if self.comm.rank == 0:
                    cublas_mmm(alpha, a.data, opa, b.data, opb, beta, c0.data)
                c0.redist(c)
                return c
            1 / 0

        if symmetric:
            if opa == 'N':
                assert opb == 'C' or opb == 'T' \
                    and np.issubdtype(a.dtype, np.floating)
                if a is b:
                    blas.gpu_r2k(0.5 * alpha,
                                 a.data,
                                 a.data,
                                 beta,
                                 c.data)
                else:
                    if beta == 1.0 and a.shape[1] == 0:
                        return
                    if c.data.size > 0:
                        assert beta in [0.0, 1.0]
                        # CuPy doesn't have dsyrk, so we roll our own:
                        blas.gpu_r2k(0.5 * alpha,
                                     a.data,
                                     b.data,
                                     beta,
                                     c.data)
            else:
                1 / 0
                assert opa == 'C' and opb == 'N'
                assert a is not b
                raise NotImplementedError
                blas.gpu_r2k(0.5 * alpha, a.data, b.data, beta, c.data, 'n')

        else:
            cublas_mmm(alpha, a.data, opa, b.data, opb, beta, c.data)

    def add_hermitian_conjugate(self,
                                a: Matrix,
                                scale: float) -> None:
        # Quick'n'dirty hack:
        b = a.to_cpu()
        b.add_hermitian_conjugate(scale)
        a.data[:] = b.to_xp(cp).data

    def eigh_serial(self,
                    H: Matrix,
                    S: Matrix | None,
                    eigs: np.ndarray,
                    cc: bool = False,
                    limit: int | None = None) -> None:
        assert isinstance(H.data, cp.ndarray)
        if cc and np.issubdtype(H.dtype, np.complexfloating):
            cp.negative(H.data.imag, H.data.imag)
        # Handle generalized eigenproblem
        if S is not None:
            S.invcholesky()
            H.tril2full()
            limit = limit or self.shape[0]
            eigs[:] = self.eighl(H, S)[:limit]
            H.data[:limit] = H.data.T[:limit].copy()
            return

        # TODO some way for the caller to specify options/backend
        diagonalizer, options = suggest_diagonalizer(H)
        options.uplo = 'L'
        options.inplace = True
        eigs[:], _ = diagonalizer.eigh(H, options)

    def eighl(self, H, L):
        """
        :::

           ~      †   ~~   ~         †~
           H = LHL ,  HC = CΛ,  C = L C.
        """
        assert self.comm.size == 1
        tmp = H.new()
        self.multiply(1.0, L, 'N', H, 'N', 0.0, tmp)
        self.multiply(1.0, tmp, 'N', L, 'C', 0.0, H, symmetric=True)

        diagonalizer, options = suggest_diagonalizer(H)
        options.inplace = False
        options.uplo = 'L'
        eig_M, Ct = diagonalizer.eigh(H, options)
        Ct_MM = Ct.data.T

        assert Ct_MM.flags.f_contiguous
        Ct = H.new(data=Ct_MM.T)
        self.multiply(1.0, L, 'C', Ct, 'T', 0.0, H)
        # H.complex_conjugate()
        return eig_M


def mmm_nn(m1, m2, m3, alpha, beta, mmm):
    """Parallel matrix-matrix multiplication.

    :::

        m  <- αm m + βm
         3      1 2    3
    """
    comm = m1.dist.comm
    buf1 = m2.data
    xp = m1.xp

    N = m1.shape[1]
    assert N == m2.shape[0], f'{N}, {m2.shape[0]}'
    n = (N + comm.size - 1) // comm.size

    for r in range(comm.size):
        if r == 0:
            # Buffers...
            buf2 = xp.empty((n, buf1.shape[1]), dtype=buf1.dtype)

        rrequest = None
        srequest = None
        if r < comm.size - 1:
            rrank = (comm.rank + r + 1) % comm.size
            rn1 = min(rrank * n, N)
            rn2 = min(rn1 + n, N)
            if rn2 > rn1:
                rrequest = comm.receive(buf2[:rn2 - rn1], rrank, 21, False)
            srank = (comm.rank - r - 1) % comm.size
            if len(m2.data) > 0:
                srequest = comm.send(m2.data, srank, 21, False)

        r0 = (comm.rank + r) % comm.size
        n1 = min(r0 * n, N)
        n2 = min(n1 + n, N)
        # Contiguity...
        mmm(alpha, m1.data[:, n1:n2], 'N', buf1[:n2 - n1], 'N', beta, m3.data)

        beta = 1.0

        if r == 0:
            # Buffers...
            buf1 = xp.empty_like(buf2)

        buf1, buf2 = buf2, buf1

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

    return m3


def mmm_nc_sym(a, b, out, alpha, mmm):
    """Symmetric parallel matrix-matrix multiplication.

    :::

                †
        c <- αab + c

    This function utilizes the fact that c is symmetric, s.t.:
                       †     †
        c <- 0.5 * (αab + αba) + c
    Only lower half of c is updated.
    """
    comm = a.dist.comm
    M, N = b.shape
    m = (M + comm.size - 1) // comm.size
    mym = len(b.data)
    xp = a.xp

    # Buffers...
    buf1 = xp.empty((m, N), dtype=a.dtype)
    buf2 = xp.empty((m, N), dtype=a.dtype)
    half = comm.size // 2
    aa = a.data
    bb = b.data

    for r in range(half + 1):
        rrequest = None
        srequest = None

        if r < half:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            skip = (comm.size % 2 == 0 and r == half - 1)
            m1 = min(rrank * m, M)
            m2 = min(m1 + m, M)
            if not (skip and comm.rank < half) and m2 > m1:
                rrequest = comm.receive(buf1[:m2 - m1], rrank, 11, False)
            if not (skip and comm.rank >= half) and mym > 0:
                srequest = comm.send(b.data, srank, 11, False)

        if not (comm.size % 2 == 0 and r == half and comm.rank < half):
            m1 = min(((comm.rank - r) % comm.size) * m, M)
            m2 = min(m1 + m, M)
            if r == 0:
                # symmmmmmmmmmmmmmmmmmmmmmetricccccccccccccccc
                # Contiguity...
                mmm(alpha, aa, 'N', bb, 'C', 1.0, out.data[:, m1:m2])
            else:
                beta = 1.0 if r <= comm.rank else 0.0
                mmm(alpha, aa, 'N', buf2[:m2 - m1], 'C',
                    beta, out.data[:, m1:m2])
            # out.data[:, m1:m2] = m12.data[:, :m2 - m1]

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        bb = buf1
        buf1, buf2 = buf2, buf1

    requests = []
    blocks = []
    nrows = (comm.size - 1) // 2
    for row in range(nrows):
        for column in range(comm.size - nrows + row, comm.size):
            if comm.rank == row:
                m1 = min(column * m, M)
                m2 = min(m1 + m, M)
                if mym > 0 and m2 > m1:
                    requests.append(
                        comm.send(out.data[:, m1:m2].T.conj().copy(),
                                  column, 12, False))
            elif comm.rank == column:
                m1 = min(row * m, M)
                m2 = min(m1 + m, M)
                if mym > 0 and m2 > m1:
                    block = xp.empty((mym, m2 - m1), out.dtype)
                    blocks.append((m1, m2, block))
                    requests.append(comm.receive(block, row, 12, False))

    comm.waitall(requests)
    for m1, m2, block in blocks:
        out.data[:, m1:m2] += block

    return out


def mmm_nc(a, b, out, alpha, beta, mmm):
    """Parallel matrix-matrix multiplication.

    :::

                †
        c <- αab  + βc
    """
    comm = a.dist.comm
    M, N = b.shape
    m = (M + comm.size - 1) // comm.size
    mym = len(b.data)
    xp = a.xp

    # Nasty buffers
    buf1 = xp.empty((m, N), dtype=a.dtype)
    buf2 = xp.empty((m, N), dtype=a.dtype)
    aa = a.data
    bb = b.data

    for r in range(comm.size):
        rrequest = None
        srequest = None

        if r < comm.size - 1:
            srank = (comm.rank + r + 1) % comm.size
            rrank = (comm.rank - r - 1) % comm.size
            m1 = min(rrank * m, M)
            m2 = min(m1 + m, M)
            if m2 > m1:
                rrequest = comm.receive(buf1[:m2 - m1], rrank, 11, False)
            if mym > 0:
                srequest = comm.send(b.data, srank, 11, False)

        m1 = min(((comm.rank - r) % comm.size) * m, M)
        m2 = min(m1 + m, M)
        # symmmmmmmmmmmmmmmmmmmmmmetricccccccccccccccc ??
        mmm(alpha, aa, 'N', bb[:m2 - m1], 'C', beta, out.data[:, m1:m2])

        if rrequest:
            comm.wait(rrequest)
        if srequest:
            comm.wait(srequest)

        bb = buf1
        buf1, buf2 = buf2, buf1

    return out
