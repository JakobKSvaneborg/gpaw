import numpy as np
from gpaw.directmin.tools import d_matrix, expm_ed


class SkewHermitian:
    """
    Class for working with skew-Hermitian matrices A (i.e., A^\\dagger = -A).

    Only the independent upper-triangular elements are stored
    in a 1D vector (`self.data`).

    Attributes
    ----------
    ndim : int
        Dimension of the square matrix (number of rows/columns).
    dtype : type
        Either float or complex.
        - float → real skew-symmetric matrices
        - complex → skew-Hermitian matrices
    data : ndarray
        1D array storing the independent upper-triangular
        entries of the matrix.
    representation : str
        Currently only "full" is supported, meaning we store all upper-triangular entries.
    """

    def __init__(self, ndim: int, dtype: type,
                 data: np.ndarray = None, representation="full"):
        """
        Initialize a SkewHermitian object.

        Parameters
        ----------
        ndim : int
            Dimension of the full matrix.
        dtype : type
            Either float or complex.
        data : np.ndarray
            1D vector of independent parameters (upper-triangular entries).
        representation : str
            Only 'full' is currently implemented.
        """

        self.ndim = ndim
        self._dtype = dtype

        # Cached values: recomputed only when data changes
        self._data = None
        self._evecs = None      # eigenvectors of i*A
        self._evals = None      # eigenvalues of i*A
        self._rotation_mat = None  # U = exp(A)
        self._representation = "full"

        # Indices of the independent parameters:
        # - if real → strictly upper triangle
        # - if complex → include diagonal
        if self._dtype == float:
            self.ind_up = np.triu_indices(self.ndim, 1)
        elif self._dtype == complex:
            self.ind_up = np.triu_indices(self.ndim)

        # Number of independent parameters
        self._len = len(self.ind_up[0])

        # Assign initial data (if provided)
        self.data = data
        assert representation == "full"

    # ------------------------
    # Properties
    # ------------------------
    @property
    def data(self):
        """Return the parameter vector (upper-triangular entries)."""
        return self._data

    @data.setter
    def data(self, array):
        """
        Set the parameter vector (upper-triangular entries).
        Resets cached eigenvectors, eigenvalues, and rotation matrix.
        """
        if isinstance(array, np.ndarray):
            # Must be 1D vector of correct length
            assert len(array.shape) == 1
            assert len(array) == self._len

        if array is not None:
            assert self.dtype == array.dtype

        self._data = array

        # Invalidate cached values
        self._evecs = None
        self._evals = None
        self._rotation_mat = None

    @property
    def dtype(self):
        """Return matrix data type (float or complex)."""
        return self._dtype

    @property
    def representation(self):
        """Return storage representation (always 'full')."""
        return self._representation

    # ------------------------
    # Eigen-decomposition and rotation
    # ------------------------
    @property
    def evecs(self):
        """
        Eigenvectors of i*A.

        Returns
        -------
        evecs : ndarray
            Eigenvectors of i*A
        """
        if self._evecs is None:
            # Convert vector to full skew-Hermitian matrix
            a_mat = vec2skewmat(self.data, self.ndim, self.ind_up, self.dtype)
            # Compute matrix exponential, eigenvectors, and eigenvalues
            self._rotation_mat, self._evecs, self._evals = expm_ed(
                a_mat, evalevec=True
            )
        return self._evecs

    @property
    def evals(self):
        """
        Eigenvalues of i*A.

        Returns
        -------
        evals : ndarray
            Eigenvalues of i*A
        """
        if self._evecs is None:  # triggers only if not already computed
            a_mat = vec2skewmat(self.data, self.ndim, self.ind_up, self.dtype)
            self._rotation_mat, self._evecs, self._evals = expm_ed(
                a_mat, evalevec=True
            )
        return self._evals

    @property
    def rotation_mat(self):
        """
        The unitary rotation matrix U = exp(A).

        Returns
        -------
        u_nn : ndarray
            Full N×N unitary matrix.
        """
        if self._data is None:
            return None
        elif self._rotation_mat is None:  # compute only once
            a_mat = vec2skewmat(self.data, self.ndim, self.ind_up, self.dtype)
            self._rotation_mat, self._evecs, self._evals = expm_ed(
                a_mat, evalevec=True
            )
        return self._rotation_mat

    # ------------------------
    # Operator overloads
    # ------------------------
    def __add__(self, other):
        """
        Allow A + B for SkewHermitian objects or numpy arrays.

        Returns
        -------
        new : SkewHermitian
            New SkewHermitian object with data = self.data + other.data
        """
        new = SkewHermitian(
            self.ndim, self.dtype, representation=self._representation
        )

        if isinstance(other, np.ndarray):
            new.data = self.data + other
            return new

        # Otherwise, other must be another SkewHermitian
        assert self.ndim == other.ndim
        assert self._representation == other.representation
        assert self.dtype == other.dtype

        new.data = self.data + other.data
        return new

    def __sub__(self, other):
        """
        Allow A - B for SkewHermitian objects or numpy arrays.

        Returns
        -------
        new : SkewHermitian
            New SkewHermitian object with data = self.data - other.data
        """
        new = SkewHermitian(
            self.ndim, self.dtype, representation=self._representation
        )

        if isinstance(other, np.ndarray):
            new.data = self.data - other
            return new

        # Otherwise, other must be another SkewHermitian
        assert self.ndim == other.ndim
        assert self._representation == other.representation
        assert self.dtype == other.dtype

        new.data = self.data - other.data
        return new

    # ------------------------
    # Gradient calculation
    # ------------------------
    def calc_gradient(self, h_nn):
        """
        Compute gradient with respect to the skew-Hermitian parameters given a matrix h_nn.

        Parameters
        ----------
        h_nn : ndarray
            Hamiltonian-like matrix of same dimension as A.

        Returns
        -------
        grad : ndarray
            1D vector (same shape as self.data) containing gradient values.
        """
        # Step 1: eigen-decomposition of i*A
        evecs = self.evecs
        evals = self.evals

        # Step 2: transform h_nn into eigenbasis of i*A
        g_mat = evecs.T.conj() @ h_nn @ evecs

        # Step 3: Weight matrix in the eigenbasis by D, which encodes
        # the derivative of the matrix exponential:
        # D_{ij} = i * (exp(-i*(omega_i - omega_j)) - 1) / (omega_i - omega_j),  with D_{ii} = 1
        # This corresponds to the derivative of exp(A) w.r.t. the skew-Hermitian parameters
        g_mat = g_mat * d_matrix(evals)

        # Step 4: transform back to original basis
        g_mat = evecs @ g_mat @ evecs.T.conj()

        # Step 5: fix diagonal scaling (skew-Hermitian structure)
        for i in range(g_mat.shape[0]):
            g_mat[i, i] *= 0.5

        # Step 6: if real dtype, discard imaginary part
        if self.dtype == float:
            g_mat = g_mat.real

        # Step 7: return only the independent upper-triangular elements
        return 2.0 * g_mat[self.ind_up]


# ------------------------
# Utility functions
# ------------------------
def vec2skewmat(a_vec, dim, ind_up, dtype):
    """
    Convert parameter vector to full skew-Hermitian matrix.

    Parameters
    ----------
    a_vec : ndarray
        1D vector of independent upper-triangular entries.
    dim : int
        Dimension of the full matrix.
    ind_up : tuple
        Indices of the upper-triangular elements.
    dtype : type
        float or complex.

    Returns
    -------
    a_mat : ndarray
        Full skew-Hermitian matrix with A^\\dagger = -A.
    """
    a_mat = np.zeros((dim, dim), dtype=dtype)
    a_mat[ind_up] = a_vec
    a_mat -= a_mat.T.conj()  # enforce skew-Hermitian property
    np.fill_diagonal(a_mat, a_mat.diagonal() * 0.5)  # fix diagonal factor
    return a_mat


def random_a(shape, dtype):
    """
    Generate random parameter vector for skew-Hermitian matrices.

    Parameters
    ----------
    shape : tuple
        Shape of the vector.
    dtype : type
        float or complex.

    Returns
    -------
    a : ndarray
        Random vector of given dtype.
    """
    a = np.random.random_sample(shape)
    if dtype == complex:
        a = a.astype(complex)
        a += 1.0j * np.random.random_sample(shape)
    return a
