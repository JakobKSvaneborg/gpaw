from typing import List, Tuple

import numpy as np
from ase.units import Ha

from gpaw.mpi import world
from gpaw.typing import Array1D, Array2D, Array3D, ArrayLike2D


class SpinDirectionConstraint:
    def __init__(self, constraint):
        self.penalty = constraint.pop('penalty')
        self.u_v = u_v

    def _tuple(self):
        # Tests use this method to compare to expected values
        return (self.penalty, self.u_v)

    def calculate(self, setup, M_vii):
        N0_q = setup.N0_q
        l_j = setup.l_j

        magmom_v = np.zeros(3)
        dHL_vii = np.zeros_like(M_vii)

        nj = len(l_j)
        i1 = slice(0, 0)
        for j1, l1 in enumerate(l_j):
                i1 = slice(i1.stop, i1.stop + 2 * l1 + 1)

                i2 = slice(0, 0)
                for j2, l2 in enumerate(l_j):
                    i2 = slice(i2.stop, i2.stop + 2 * l2 + 1)
                    if not l1 == l2:
                        continue
                    N0 = N0_q[(j2 + j1 * nj - j1 * (j1 + 1) // 2
                               if j1 < j2 else
                               j1 + j2 * nj - j2 * (j2 + 1) // 2)]

                    magmom_v += np.sum(M_vii[:, i1, i2], axis=(1, 2)) * N0
                    dHL_vii[:, i1, i2] += np.eye(2 * l1 + 1) * N0

        #global counter
        if world.rank == 0:
            u_v = np.array([0, 1, 0])
        elif world.rank == 1:
            u_v = np.array([0, 0, 1])

        # eL = constraining_field(magmom_v, dHL_vii, self.penalty, self.u_v)
        eL = constraining_field(magmom_v, dHL_vii, self.penalty, u_v)

        return eL, dHL_vii

    def descriptions(self):
        yield f'cDFT Penalty: {self.penalty * Ha},  # eV'


def constraining_field(smm_v: Array1D,
                       dHL_vii: Array3D,
                       penalty: float,
                       u_v: Array1D) -> float:

    # eL = penalty * (np.dot(smm_v, smm_v) - np.dot(u_v, smm_v)**2)
    
    dHL_vii[0] *= (1 - u_v[0]**2) * smm_v[0] - u_v[0] * (
        u_v[1] * smm_v[1] + u_v[2] * smm_v[2])
    dHL_vii[1] *= (1 - u_v[1]**2) * smm_v[1] - u_v[1] * (
        u_v[2] * smm_v[2] + u_v[0] * smm_v[0])
    dHL_vii[2] *= (1 - u_v[2]**2) * smm_v[2] - u_v[2] * (
        u_v[0] * smm_v[0] + u_v[1] * smm_v[1])

    dHL_vii *= 2 * penalty

    return 0.
