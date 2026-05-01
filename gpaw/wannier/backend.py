"""GPAW backend for ASE's Wannier function module.

This module provides :class:`GPAWBackend`, the concrete implementation
of :class:`ase.dft.wannierbackend.WannierBackend` for GPAW.

Usage::

    from gpaw import GPAW
    from gpaw.wannier.backend import GPAWBackend
    from ase.dft.wannier import Wannier

    calc = GPAW('saved.gpw')
    backend = GPAWBackend(calc)
    wan = Wannier(nwannier=4, backend=backend)
    wan.localize()
"""

from __future__ import annotations

import numpy as np
from ase.dft.wannierbackend import WannierBackend


class GPAWBackend(WannierBackend):
    """Wannier backend for GPAW calculations.

    Parameters
    ----------
    calc:
        A converged GPAW calculator instance (typically loaded from a
        ``.gpw`` file).
    """

    def __init__(self, calc):
        self._calc = calc

    def get_calcdata(self):
        from ase.dft.wannier import get_calcdata
        return get_calcdata(self._calc)

    def get_wannier_localization_matrix(self, nbands, dirG, kpoint,
                                        nextkpoint, G_I, spin):
        from gpaw.new.wannier import get_wannier_integrals
        dft = self._calc.dft
        grid = dft.density.nt_sR.desc
        k_kc = dft.ibzwfs.ibz.bz.kpt_Kc
        G_c = k_kc[nextkpoint] - k_kc[kpoint] - G_I
        return get_wannier_integrals(dft.ibzwfs, grid,
                                     spin, kpoint, nextkpoint,
                                     G_c, nbands)

    def get_wave_function_on_grid(self, band, kpt, spin):
        return self._calc.get_pseudo_wave_function(
            band, kpt, spin, pad=True)

    def get_grid_dimensions(self):
        return self._calc.get_number_of_grid_points()

    def get_initial_projections(self, initialwannier, kpointgrid,
                                fixedstates, edf, spin, nbands):
        from gpaw.new.wannier import initial_wannier
        return initial_wannier(self._calc.dft.ibzwfs,
                               initialwannier, kpointgrid,
                               fixedstates, edf, spin, nbands)
