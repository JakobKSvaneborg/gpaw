from ase.units import Bohr, Ha
from gpaw.new.etdm.utils import calc_self_hartree_derivative
from gpaw.new.etdm.utils import self_hartree_paw

from gpaw.utilities.tools import cutoff2gridspacing

from gpaw.new.etdm.objfunc_etdm import ObjectiveFunctionETDM
import numpy as np


class EdmistonRuedenberg(ObjectiveFunctionETDM):
    def __init__(self, ibzwfs, loctype="pseudo-paw", indices="all"):

        dtype, nkps = (
            ibzwfs.dtype,
            ibzwfs.nspins * len(ibzwfs.ibz),
        )

        assert nkps == 1

        if indices == "all":
            ndim = ibzwfs.nbands
            self._indices = range(ndim)
        elif indices == "occupied":
            f_n = ibzwfs.get_all_eigs_and_occs(broadcast=True)[1]
            if ibzwfs.domain_comm.rank != 0:
                f_n = np.zeros(ibzwfs.nbands)
            ibzwfs.domain_comm.broadcast(f_n, 0)
            self._indices = f_n > 1.0e-5
            ndim = sum(self._indices)
        elif indices == "virtual":
            f_n = ibzwfs.get_eigs_and_occs(0, 0)[1]
            if ibzwfs.domain_comm.rank != 0:
                f_n = np.zeros(ibzwfs.nbands)
            ibzwfs.domain_comm.broadcast(f_n, 0)
            self._indices = f_n < 1.0e-5
            ndim = sum(self._indices)
        else:
            raise NotImplementedError

        super().__init__(ndim, dtype, nkps)
        self._ibzwfs = ibzwfs
        self._rpsi_unX = []  # r is for reference
        self._rP_uani = []  # r is for reference
        self._dtype = ibzwfs.dtype
        self._loc_type = loctype
        assert (
            loctype == "pseudo" or loctype == "paw" or loctype == "pseudo-paw"
        )

        for wfs in ibzwfs:
            self._rpsi_unX.append(wfs.psit_nX.copy())
            P2_ani = wfs.P_ani.copy()
            for a, P_ni in wfs.P_ani.items():
                P2_ani[a] = P_ni.copy()
            self._rP_uani.append(P2_ani)

    @property
    def kpt_comm(self):
        return self._ibzwfs.kpt_comm

    def _calc_obf_value_and_matrix_elements(self):
        """Calculate value of the objective function (energy)
        and hamiltonian matrix elements
        at self._a_vec_u

        Parameters
        ----------

        Returns
        -------
        energy : float
        h_unn : ndarray
            shape = (n_kps, ndim, ndim)
        """

        self.rotate_wfs()

        energy = 0
        h_unn = []

        for wfs in self._ibzwfs:
            psit_nX = wfs.psit_nX
            h_nn = np.zeros(shape=(self._ndim, self._ndim), dtype=wfs.dtype)
            if "pseudo" in self._loc_type:
                grid_spacing = (
                    cutoff2gridspacing(3 * psit_nX.desc.ecut * Ha) / Bohr
                )
                derivative_nG = calc_self_hartree_derivative(
                    psit_nX, grid_spacing
                )
                h_nn += derivative_nG.integrate(psit_nX).T[
                    np.ix_(self._indices, self._indices)
                ]
                energy += np.diag(h_nn).sum() * 0.5
            if "paw" in self._loc_type:
                h_paw_nn = self_hartree_paw(wfs, wfs.setups, None, None)
                h_paw_nn = h_paw_nn[np.ix_(self._indices, self._indices)]
                energy += np.diag(h_paw_nn).sum() * 0.5
                h_nn += h_paw_nn

            h_unn.append(h_nn)

        return energy.real, np.array(h_unn)

    def rotate_wfs(self):

        for a, wfs, rpsit_nX, rP_ani in zip(
            self._a_vec_u,
            self._ibzwfs,
            self._rpsi_unX,
            self._rP_uani,
        ):
            u_nn = a.rotation_mat

            # rotate wave functions; it can be a separate function
            wfs.psit_nX.data[self._indices] = np.tensordot(
                u_nn.T, rpsit_nX.data[self._indices], axes=1
            )
            for a, P_ni in wfs.P_ani.items():
                P_ni[self._indices] = u_nn.T @ rP_ani[a][self._indices]


class EdmistonRuedenbergUpdateRef(EdmistonRuedenberg):
    def __init__(self, ibzwfs, loctype="pseudo-paw", indices="all"):
        super().__init__(ibzwfs, loctype, indices)

    def rotate_wfs(self):

        for a, wfs in zip(self._a_vec_u, self._ibzwfs):
            u_nn = a.rotation_mat

            # rotate wave functions; it can be a separate function
            wfs.psit_nX.data[self._indices] = np.tensordot(
                u_nn.T, wfs.psit_nX.data[self._indices], axes=1
            )
            for a, P_ni in wfs.P_ani.items():
                P_ni[self._indices] = u_nn.T @ P_ni[self._indices]

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
                2 * (h_nn - h_nn.T.conj())[a.ind_up]
                for (a, h_nn) in zip(self.a_vec_u, h_unn)
            ]
        )
        return energy, gradient
