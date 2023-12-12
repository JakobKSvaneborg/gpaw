from __future__ import annotations

import numpy as np
from ase.units import Ha

from gpaw.typing import Array1D, Array2D, ArrayLike1D
from gpaw.core import UGArray
from gpaw.new.density import Density


def create_external_potential(params: dict) -> ExternalPotential:
    if not params:
        return ExternalPotential()
    assert params['name'] == 'BField', len(params) == 2
    return BField(np.asarray(params['field']) / Ha)


class ExternalPotential:
    def update_potential(self,
                         vt_sR: UGArray,
                         density) -> float:
        return 0.0

    def add_paw_correction(self, Delta_p: Array1D, dH_sp: Array2D) -> float:
        return 0.0


class BField(ExternalPotential):
    def __init__(self, field: ArrayLike1D):
        """Constant magnetic field.

        field:
            B-field vector in units of Ha/bohr-magnoton.
        """
        self.field_v = np.array(field)
        assert self.field_v.shape == (3,)

    def update_potential(self,
                         vt_sR: UGArray,
                         density: Density) -> float:
        magmom_v, _ = density.calculate_magnetic_moments()
        eext = -self.field_v @ magmom_v
        ncomponents = len(vt_sR)
        if ncomponents == 2:
            assert (self.field_v[:2] == 0.0).all()
            vt_sR.data[0] -= self.field_v[2]
            vt_sR.data[1] += self.field_v[2]
        elif ncomponents == 4:
            vt_sR.data[1:] = -self.field_v.reshape((3, 1, 1, 1))
        else:
            1 / 0
        return eext

    def add_paw_correction(self, Delta_p: Array1D, dH_sp: Array2D) -> float:
        if len(dH_sp) == 2:
            c = (4 * np.pi)**0.5 * self.field_v[2]
            dH_sp[0] -= c * Delta_p
            dH_sp[1] += c * Delta_p
        else:
            c_vp = (4 * np.pi)**0.5 * self.field_v[:, np.newaxis]
            dH_sp[1:] -= c_vp * Delta_p
        return 0.0
