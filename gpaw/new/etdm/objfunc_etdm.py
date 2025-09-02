import numpy as np
from .skewherm_matrix import SkewHermitian
from abc import ABC, abstractmethod


class ObjectiveFunctionETDM(ABC):
    def __init__(self, ndim: int, dtype: type, nkps: int):

        self._a_vec_u = [
            SkewHermitian(ndim, dtype, representation="full")
            for _ in range(nkps)
        ]

        self._energy = None
        self._gradient = None
        self._ndim = ndim
        self._dtype = dtype
        self._nkps = nkps

    @property
    def a_vec_u(self):
        return self._a_vec_u

    @a_vec_u.setter
    def a_vec_u(self, a_u):
        for u, a in enumerate(self._a_vec_u):
            a.data = a_u[u]
        self._energy = None
        self._gradient = None

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

    def _calc_energy_and_gradient(self, a_u: np.ndarray = None):
        """Calculate value of the objective function (energy) and gradient at a_u

        Parameters
        ----------
        a_u

        Returns
        -------

        """
        if a_u is not None:
            self.a_vec_u = a_u

        energy, h_unn = self._calc_obf_value_and_matrix_elements()
        gradient = np.array(
            [
                a.calc_gradient(h_nn - h_nn.T.conj())
                for (a, h_nn) in zip(self.a_vec_u, h_unn)
            ]
        )
        return energy, gradient

    @abstractmethod
    def _calc_obf_value_and_matrix_elements(self):
        """Calculate value of the objective function (energy) and hamiltonian matrix elements
        at self._a_vec_u

        Parameters
        ----------

        Returns
        -------
        energy : float
        h_unn : ndarray
            shape = (n_kps, ndim, ndim)
        """
        pass

