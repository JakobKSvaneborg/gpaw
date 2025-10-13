import numpy as np
from ase.parallel import parprint as print
from .searchdir import LBFGS


class ETDM:
    """
    ETDM optimizer class.

    Attributes
    ----------
    a_u : ndarray
        Current parameter vector representing the independent elements
        of the skew-Hermitian matrices for all k-points.
    """

    def __init__(
        self,
        objfunc,
        a_u_init: np.ndarray,
        maxiter=100,
        tolerance=5.0e-4,
        update_ref=False,
    ):
        """
        Initialize the ETDM optimizer.

        Parameters
        ----------
        objfunc : ObjectiveFunctionETDM
            The objective function to optimize. Must provide `.energy`
            and `.gradient` and accept updates via `a_vec_u`.
        a_u_init : ndarray
            Initial parameter vector for the optimization.
        maxiter : int
            Maximum number of optimization iterations.
        tolerance : float
            Convergence tolerance for the gradient norm.
        update_ref : bool
            Determines how the iterative update is applied:
                - True:    C_{j+1} = C_j exp(A_j)
                - False:   C_{j+1} = C_0 exp(sum_{0}^{j} A_j)
        """

        self.objfunc = objfunc

        # Initialize LBFGS search direction algorithm
        # `searchdir_algo` keeps track of the search direction `p_u`
        # and performs the quasi-Newton update.
        self.searchdir_algo = LBFGS(
            a_u_init.shape, objfunc.kpt_comm, objfunc._dtype
        )

        self.iter = 0                   # Iteration counter
        self._tolerance = tolerance     # Convergence threshold
        self._max_iter = maxiter        # Maximum iterations allowed
        self._update_ref = update_ref   # Flag for update type

        self._a_u = a_u_init            # Current parameters
        self._energy = None             # Cached energy
        self._gradient = None           # Cached gradient
        self._error = None              # Cached max gradient norm
        self._is_converged = False      # Convergence flag

    def optimize(self):
        """
        Main optimization loop.

        Iteratively updates `a_u` using LBFGS search directions until
        the gradient norm is below tolerance or maximum iterations are reached.
        """
        while (not self.is_converged) and self.iter < self._max_iter:
            # Update the search direction using LBFGS
            self.searchdir_algo.update(self.a_u, self.gradient)

            # Move parameters along the search direction
            self.move()

            # Increment iteration counter
            self.iter += 1

            # Periodically print the status
            if self.iter % 20 == 0:
                print(self.iter, self.energy, self.error)

    def move(self):
        """
        Apply a step along the current search direction.

        This modifies `a_u` either as a relative update (update_ref=True)
        or accumulated from the initial vector (update_ref=False).
        """
        p_u = self.searchdir_algo.search_dir  # Current LBFGS search direction

        # Compute norm of search direction across all k-points
        strength = np.sum(p_u.conj() * p_u)
        strength = self.objfunc.kpt_comm.sum(strength.real) ** 0.5

        # Scale step to prevent too large updates
        alpha = np.minimum(0.25 / strength, 1.0)
        p_u[:] = alpha * p_u  # Scale the search direction

        # Update the parameters
        if self._update_ref:
            # Relative update: C_{j+1} = C_j exp(A_j)
            self.a_u = p_u
        else:
            # Accumulated update: C_{j+1} = C_0 exp(sum_0^j A_j)
            self.a_u += p_u

    @property
    def a_u(self):
        """
        Current parameter vector.

        Returns
        -------
        ndarray
        """
        return self._a_u

    @a_u.setter
    def a_u(self, x):
        """
        Update the parameter vector and invalidate cached values.

        Parameters
        ----------
        x : ndarray
            New parameter vector
        """
        self._a_u = x
        self._energy = None
        self._gradient = None
        self._error = None
        self._is_converged = None

    @property
    def energy(self):
        """
        Current energy of the objective function, computed lazily.

        Returns
        -------
        float
        """
        if self._energy is None:
            self._energy, self._gradient = self._calc_energy_and_gradient()
        return self._energy

    @property
    def gradient(self):
        """
        Current gradient of the objective function w.r.t. `a_u`, computed lazily.

        Returns
        -------
        ndarray
        """
        if self._gradient is None:
            self._energy, self._gradient = self._calc_energy_and_gradient()
        return self._gradient

    @property
    def error(self):
        """
        Maximum absolute value of the gradient (used for convergence check).

        Returns
        -------
        float
        """
        if self._error is None:
            self._error = np.max(np.abs(self.gradient))
        return self._error

    @property
    def is_converged(self):
        """
        Check if the optimization has converged based on gradient norm.

        Returns
        -------
        bool
        """
        if self.error < self._tolerance:
            self._is_converged = True
        else:
            self._is_converged = False
        return self._is_converged

    def _calc_energy_and_gradient(self):
        """
        Compute energy and gradient from the objective function.

        Updates `objfunc.a_vec_u` to the current `a_u` parameters
        and retrieves energy and gradient. This ensures consistency
        between the optimizer and the objective function.

        Returns
        -------
        energy : float
        gradient : ndarray
        """
        self.objfunc.a_vec_u = self.a_u
        return self.objfunc.energy, self.objfunc.gradient
