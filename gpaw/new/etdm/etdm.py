import numpy as np
from ase.parallel import parprint as print

from .searchdir import LBFGS


class ETDM:
    """
    Attributes
    ----------
    a_u : ndarray
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

        Parameters
        ----------
        objfunc
        maxiter
        maxstepxst
        g_tol
        update_ref : bool
            if True then the iterations are:
                math:`C_{j+1} = C_{j} exp(A_j)`
            otherwise:
                math:`C_{j+1} = C_{0} exp(\sum_{0}^{j} A_j)`
        """

        self.objfunc = objfunc
        self.searchdir_algo = LBFGS(
            a_u_init.shape, objfunc.kpt_comm, objfunc._dtype
        )
        self.iter = 0
        self._tolerance = tolerance
        self._max_iter = maxiter
        self._update_ref = update_ref

        self._a_u = a_u_init
        self._energy = None
        self._gradient = None
        self._error = None
        self._is_converged = False

    def optimize(self):

        print(self.iter, self.energy, self.error)

        while (not self.is_converged) and self.iter < self._max_iter:
            self.searchdir_algo.update(self.a_u, self.gradient)
            self.move()
            self.iter += 1
            if self.iter % 20 == 0:
                print(self.iter, self.energy, self.error)

    def move(self):
        p_u = self.searchdir_algo.search_dir
        strength = np.sum(p_u.conj() * p_u)
        strength = self.objfunc.kpt_comm.sum(strength.real) ** 0.5
        alpha = np.minimum(0.25 / strength, 1.0)
        # since we have a reference here we
        # also modify self.searchdir_algo.search_dir and
        # this is what we need
        p_u[:] = alpha * p_u
        if self._update_ref:
            self.a_u = p_u
        else:
            self.a_u += p_u

    @property
    def a_u(self):
        return self._a_u

    @a_u.setter
    def a_u(self, x):
        self._a_u = x
        self._energy = None
        self._gradient = None
        self._error = None
        self._is_converged = None

    @property
    def energy(self):
        if self._energy is None:
            self._energy, self._gradient = self._calc_energy_and_gradient()
        return self._energy

    @property
    def gradient(self):
        if self._gradient is None:
            self._energy, self._gradient = self._calc_energy_and_gradient()
        return self._gradient

    @property
    def error(self):
        if self._error is None:
            self._error = np.max(np.abs(self.gradient))
        return self._error

    @property
    def is_converged(self):
        if self.error < self._tolerance:
            self._is_converged = True
        else:
            self._is_converged = False
        return self._is_converged

    def _calc_energy_and_gradient(self):
        self.objfunc.a_vec_u = self.a_u
        return self.objfunc.energy, self.objfunc.gradient
