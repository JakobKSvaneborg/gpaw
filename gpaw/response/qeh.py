from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from ase.units import pi
from typing import Any, Dict, Union
from gpaw.response.df import DielectricFunction


def _sanitize_for_npz(value: Any) -> Any:
    """Sanitize a value to avoid requiring pickle when saving to npz.

    Converts problematic types that would require allow_pickle=True:
    - None -> "__none__" (numpy converts None to object arrays)
    - 0-d object arrays (e.g., np.array(None, dtype=object)) -> extracted scalar
    - Object arrays with a single element -> the element itself
    - Recursively sanitizes dicts and dict-like objects (including NpzFile)

    Parameters
    ----------
    value : Any
        The value to sanitize.

    Returns
    -------
    Any
        The sanitized value that can be saved without pickling.
    """
    if value is None:
        # numpy converts None to np.array(None, dtype=object) which needs pickle
        return "__none__"
    if isinstance(value, np.ndarray):
        # Object arrays require pickling - try to convert them
        if value.dtype == object:
            # 0-d array with single value (e.g., np.array(None, dtype=object))
            if value.ndim == 0:
                return _sanitize_for_npz(value.item())
            # 1-d array with single element
            if value.ndim == 1 and value.size == 1:
                return _sanitize_for_npz(value[0])
            # Multi-element object array - convert to list if possible
            try:
                return [_sanitize_for_npz(item) for item in value]
            except (TypeError, ValueError):
                # Can't convert, return as-is (will require pickling)
                return value
        # Non-object arrays are fine for npz
        return value
    elif isinstance(value, Mapping):
        # Handles dict, NpzFile, and other dict-like objects
        return {k: _sanitize_for_npz(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        sanitized = [_sanitize_for_npz(item) for item in value]
        return type(value)(sanitized)
    elif isinstance(value, (np.integer, np.floating)):
        # Convert numpy scalar types to Python types
        return value.item()
    elif isinstance(value, np.bool_):
        return bool(value)
    else:
        # Basic Python types (int, float, str, bool) are fine
        return value


def _desanitize_from_npz(value: Any) -> Any:
    """Convert "__none__" back to None, recursively for dicts."""
    if isinstance(value, dict):
        return {k: _desanitize_from_npz(v) for k, v in value.items()}
    elif isinstance(value, str) and value == '__none__':
        return None
    return value


try:
    from qeh.bb_calculator.chicalc import ChiCalc, QPoint
except ImportError:
    class ChiCalc():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError('qeh not installed, \
                               or is too old.')

    class QPoint():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError('qeh not installed, \
                               or is too old.')


class QEHChiCalc(ChiCalc):
    qdim = {'x': 0, 'y': 1}

    def __init__(self,
                 df: DielectricFunction,
                 qinf_rel: float = 1e-6,
                 direction: str = 'x',
                 gamma_tol: float = 1e-8):

        ''' GPAW superclass for interfacing with QEH
        building block calculations.

        Parameters
        ----------
        df : DielectricFunction
            the dielectric function calculator
        qinf_rel : float
            the position of the gamma q-point,
            relative to the first non-gamma q-point,
            necessary due to the undefined nature of
            chi_wGG in the gamma q-point.
        direction : str (either 'x' or 'y')
            the direction of the q-grid in terms of
            the reciprocal lattice vectors.
        gamma_tol: float
            tolerance for determining whether the k-grid
            contains Gamma
        '''
        if direction not in self.qdim:
            raise ValueError(
                f"direction must be 'x' or 'y', got {direction!r}")
        self.direction = direction
        self.qinf_rel = qinf_rel
        self.df = df
        self.gd = df.gs.gd
        self.L = df.gs.gd.cell_cv[2, 2]

        self.kd = df.gs.kd
        ibzk_kc = self.kd.ibzk_kc
        gamma_centered = np.any(np.linalg.norm(ibzk_kc, axis=1) < gamma_tol)
        if not gamma_centered:
            raise ValueError('Only Gamma-centered '
                             'k-point grids are supported')
        self.qdir = self.qdim[self.direction]
        self.Nk = self.kd.N_c[self.qdir]

        self.omega_w = self.df.chi0calc.wd.omega_w
        self.context = self.df.context

        super().__init__()

    def get_q_grid(self, q_max: float | None = None):
        # First get q-points on the grid

        icell_cv = self.gd.icell_cv

        # Make q-grid
        q_qc = np.zeros([self.Nk, 3], dtype=float)
        q_qc[:, self.qdir] = np.linspace(0, 1, self.Nk,
                                         endpoint=False)

        # Avoid Gamma-point
        q_qc[0] = q_qc[1] * self.qinf_rel

        q_qv = q_qc @ icell_cv * 2 * pi

        # Filter the q-points with q_max
        if q_max is not None:
            q_mask = np.linalg.norm(q_qv, axis=1) <= q_max
            q_qc = q_qc[q_mask]
            q_qv = q_qv[q_mask]

        # Make list of QPoints for calculation
        Q_q = [QPoint(q_c=q_c, q_v=q_v,
                      P_rv=self.determine_P_rv(q_c, q_max))
               for q_c, q_v in zip(q_qc, q_qv)]

        return Q_q

    def determine_P_rv(self, q_c: np.ndarray, q_max: float | None):
        """
        Determine the reciprocal space vectors P_rv that correspond
        to unfold the given q-point out of the 1st BZ
        given a q-point and the maximum q-value.

        Parameters:
            q_c (np.ndarray): The q-point in reciprocal space.
            q_max (float | None): The maximum q-value.

        Returns:
            np.ndarray: array of reciprocal space vectors P_rv.
        """
        if q_max is None:
            return np.array([[0, 0, 0]])

        icell_cv = self.gd.icell_cv
        G_v = icell_cv[self.qdir] * 2 * pi
        qc_max = q_max / np.linalg.norm(G_v)  # max |q| in crystal units
        nP = int(qc_max - q_c[self.qdir]) + 1
        if nP <= 0:
            return np.array([[0, 0, 0]])
        i_r = np.arange(nP)
        P_rv = i_r[:, None] * G_v
        return P_rv

    def get_z_grid(self):
        r = self.gd.get_grid_point_coordinates()
        return r[2, 0, 0, :].copy()

    def get_chi_wGG(self, qpoint: QPoint):
        if np.linalg.norm(qpoint.q_c) <= (2 * self.qinf_rel / self.Nk):
            chi0_dyson_eqs = self.df.get_chi0_dyson_eqs([0, 0, 0],
                                                        truncation='2D')
            qpd, chi_wGG, wblocks = chi0_dyson_eqs.rpa_density_response(
                qinf_v=qpoint.q_v, direction=qpoint.q_v)
        else:
            chi0_dyson_eqs = self.df.get_chi0_dyson_eqs(qpoint.q_c,
                                                        truncation='2D')
            qpd, chi_wGG, wblocks = chi0_dyson_eqs.rpa_density_response()

        G_Gv = qpd.get_reciprocal_vectors(add_q=False)

        return chi_wGG, G_Gv, wblocks

    def get_atoms(self):
        return self.df.gs.atoms

    def get_calc_info(self, extra_info: Union[dict, None] = None) -> dict:
        info: Dict[str, Any] = {}
        df = self.df
        gs = df.gs
        chi0_body = df.chi0calc.chi0_body_calc
        ecut = chi0_body.ecut
        if isinstance(ecut, dict):
            ecut_xy = ecut['kwargs']['ecut_xy']
            ecut_z = ecut['kwargs']['ecut_z']
            ecut = {
                'class': 'cylindrical',
                'ecut_xy': ecut_xy,
                'ecut_z': ecut_z}
        else:
            ecut = {
                'class': 'spherical',
                'ecut': ecut
            }
        info['ecut'] = ecut
        info['eta'] = chi0_body.eta
        info['truncation'] = df.truncation
        info['eshift'] = chi0_body.eshift
        info['nbands'] = chi0_body.nbands
        info['bzkpts_kc'] = gs.kd.bzk_kc
        info['nbzkpts'] = len(gs.kd.bzk_kc)
        info['kpt_info'] = str(gs.kd)
        info['unit_cell_cv'] = gs.atoms.cell

        if extra_info is not None:
            info.update(extra_info)

        # Sanitize all values to avoid requiring pickle when saving to npz
        return _sanitize_for_npz(info)
