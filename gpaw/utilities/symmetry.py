from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from gpaw import debug
from gpaw.typing import Array2D

if TYPE_CHECKING:
    from gpaw.new.symmetry import Symmetries


@cache
def totally_unimodular_matrices() -> np.ndarray:
    # Symmetry operations as matrices in 123 basis.
    # Operation is a 3x3 matrix, with possible elements -1, 0, 1, thus
    # there are 3**9 = 19683 possible matrices:
    combinations = 1 - np.indices([3] * 9)
    U_scc = combinations.reshape((3, 3, 3**9)).transpose((2, 0, 1))
    U_scc = U_scc.astype(float)
    U_scc = U_scc[abs(np.linalg.det(U_scc)) == 1.0]  # reduce to 6960
    return U_scc


def find_lattice_symmetry(cell_cv, pbc_c, tol, _backwards_compatible=False):
    """Determine list of symmetry operations."""
    U_scc = totally_unimodular_matrices()

    # The metric of the cell should be conserved after applying
    # the operation:
    metric_cc = cell_cv.dot(cell_cv.T)
    metric_scc = np.einsum('sij, jk, slk -> sil',
                           U_scc, metric_cc, U_scc,
                           optimize=True)

    if _backwards_compatible:
        # (wrong units)
        mask_s = abs(metric_scc - metric_cc).sum(2).sum(1) <= tol
    else:
        L_c = metric_cc.diagonal()**0.5
        tol_cc = np.add.outer(L_c, L_c) * tol
        err_scc = abs(metric_scc - metric_cc)
        mask_s = (err_scc <= tol_cc).all(axis=(1, 2))

    U_scc = U_scc[mask_s].astype(int)

    # Operation must not swap axes that don't have same PBC:
    pbc_cc = np.logical_xor.outer(pbc_c, pbc_c)
    mask_s = ~U_scc[:, pbc_cc].any(axis=1)
    U_scc = U_scc[mask_s]
    return U_scc


def prune_symmetries(sym: Symmetries,
                     relpos_ac: Array2D,
                     id_a: Sequence[int],
                     symmorphic: bool = True) -> Symmetries:
    """Remove symmetries that are not satisfied by the atoms."""

    if len(relpos_ac) == 0:
        return sym

    # Build lists of atom numbers for each type of atom - one
    # list for each combination of atomic number, setup type,
    # magnetic moment and basis set:
    a_ij = defaultdict(list)
    for a, id in enumerate(id_a):
        a_ij[id].append(a)

    a_j = a_ij[id_a[0]]  # just pick the first species

    def check(op_cc, ft_c):
        return sym.check_one_symmetry(relpos_ac, op_cc, ft_c, a_ij)

    # if supercell disable fractional translations:
    if not symmorphic:
        op_cc = np.identity(3, int)
        ftrans_sc = relpos_ac[a_j[1:]] - relpos_ac[a_j[0]]
        ftrans_sc -= np.rint(ftrans_sc)
        for ft_c in ftrans_sc:
            a_a = check(op_cc, ft_c)
            if a_a is not None:
                symmorphic = True
                break

    symmetries = []
    ftsymmetries = []

    # go through all possible symmetry operations
    for op_cc in sym.rotation_scc:
        # first ignore fractional translations
        a_a = check(op_cc, [0, 0, 0])
        if a_a is not None:
            symmetries.append((op_cc, [0, 0, 0], a_a))
        elif not symmorphic:
            # check fractional translations
            sposrot_ac = np.dot(relpos_ac, op_cc)
            ftrans_jc = sposrot_ac[a_j] - relpos_ac[a_j[0]]
            ftrans_jc -= np.rint(ftrans_jc)
            for ft_c in ftrans_jc:
                a_a = check(op_cc, ft_c)
                if a_a is not None:
                    ftsymmetries.append((op_cc, ft_c, a_a))

    # Add symmetry operations with fractional translations at the end:
    symmetries.extend(ftsymmetries)

    from gpaw.new.symmetry import Symmetries
    sym = Symmetries(cell=sym.cell_cv,
                     rotations=[s[0] for s in symmetries],
                     translations=[s[1] for s in symmetries],
                     atommaps=[s[2] for s in symmetries],
                     tolerance=sym.tolerance,
                     _backwards_compatible=sym._backwards_compatible)
    if debug:
        sym.check_positions(relpos_ac)

    sym.group_check()

    return sym
