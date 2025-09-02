import numpy as np
from gpaw.directmin.tools import d_matrix, expm_ed


class SkewHermitian:
    """Class for manupilation with skew-hermitian matrices

    Attributes
    __________
    ndim : int
        size of one axis
    dtype : type
        float or complex
    data : ndarray
        1-d array storing the minimal information about the skh. matrix
    representation : str
        currently supports only 'full' which means that
        data stores upper triagonal part of the matrix
        including diagonal for complex matrices
    """

    def __init__(
        self,
        ndim: int,
        dtype: type,
        data: np.ndarray = None,
        representation="full",
    ):
        """
        Parameters
        ----------
        ndim : int
            size of one matrix axis
        dtype : type
            float or complex
        data : ndarray
        representation : str
            currently supports only 'full' which means that
            data stores upper triagonal part of the matrix
            including diagonal for complex matrices
        """

        self.ndim = ndim
        self._dtype = dtype

        self._data = None
        self._evecs = None  # eigenvectors for i*data
        self._evals = None  # eigenvalues for i*data
        self._rotation_mat = None
        self._representation = "full"

        if self._dtype == float:
            self.ind_up = np.triu_indices(self.ndim, 1)
        elif self._dtype == complex:
            self.ind_up = np.triu_indices(self.ndim)

        self._len = len(self.ind_up[0])

        self.data = data
        assert representation == "full"

    @property
    def data(self):
        return self._data

    @property
    def dtype(self):
        return self._dtype

    @property
    def representation(self):
        return self._representation

    @data.setter
    def data(self, array):
        if isinstance(array, np.ndarray):
            assert len(array.shape) == 1
            assert len(array) == self._len

        if array is not None:
            assert self.dtype == array.dtype
        self._data = array
        self._evecs = None  # eigenvectors for i*data
        self._evals = None  # eigenvalues for i*data
        self._rotation_mat = None

    @property
    def evecs(self):
        if self._evecs is None:
            a_mat = vec2skewmat(self.data, self.ndim, self.ind_up, self.dtype)
            self._rotation_mat, self._evecs, self._evals = expm_ed(
                a_mat, evalevec=True
            )
        return self._evecs

    @property
    def evals(self):
        if self._evecs is None:
            a_mat = vec2skewmat(self.data, self.ndim, self.ind_up, self.dtype)
            self._rotation_mat, self._evecs, self._evals = expm_ed(
                a_mat, evalevec=True
            )
        return self._evals

    @property
    def rotation_mat(self):
        """

        Returns
        -------
        u_nn : ndarray
            unitary matrix, exp(a)
        """
        if self._data is None:
            return None
        elif self._rotation_mat is None:
            a_mat = vec2skewmat(self.data, self.ndim, self.ind_up, self.dtype)
            self._rotation_mat, self._evecs, self._evals = expm_ed(
                a_mat, evalevec=True
            )
        return self._rotation_mat

    def __add__(self, other):
        new = SkewHermitian(
            self.ndim, self.dtype, representation=self._representation
        )

        if isinstance(other, np.ndarray):
            new.data = self.data + other
            return new

        assert self.ndim == other.ndim
        assert self._representation == other.representation
        assert self.dtype == other.dtype

        new = SkewHermitian(
            self.ndim, self.dtype, representation=self._representation
        )
        new.data = self.data + other.data
        return new

    def __sub__(self, other):
        new = SkewHermitian(
            self.ndim, self.dtype, representation=self._representation
        )

        if isinstance(other, np.ndarray):
            new.data = self.data - other
            return new

        assert self.ndim == other.ndim
        assert self._representation == other.representation
        assert self.dtype == other.dtype

        new = SkewHermitian(
            self.ndim, self.dtype, representation=self._representation
        )
        new.data = self.data - other.data
        return new

    def calc_gradient(self, h_nn):

        evecs = self.evecs
        evals = self.evals

        g_mat = evecs.T.conj() @ h_nn @ evecs
        g_mat = g_mat * d_matrix(evals)
        g_mat = evecs @ g_mat @ evecs.T.conj()

        for i in range(g_mat.shape[0]):
            g_mat[i, i] *= 0.5

        if self.dtype == float:
            g_mat = g_mat.real

        return 2.0 * g_mat[self.ind_up]


def vec2skewmat(a_vec, dim, ind_up, dtype):

    a_mat = np.zeros(shape=(dim, dim), dtype=dtype)
    a_mat[ind_up] = a_vec
    a_mat -= a_mat.T.conj()
    np.fill_diagonal(a_mat, a_mat.diagonal() * 0.5)
    return a_mat


def random_a(shape, dtype):

    a = np.random.random_sample(shape)
    if dtype == complex:
        a = a.astype(complex)
        a += 1.0j * np.random.random_sample(shape)

    return a
