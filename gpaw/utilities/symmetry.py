from __future__ import annotations

import numpy as np
from collections import defaultdict
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpaw.new.symmetry import Symmetries

def find_lattice_symmetry(cell_cv, pbc_c, tol, _backwards_compatible=False):
    """Determine list of symmetry operations for a given lattice."""
    U_scc = generate_all_symmetry_matrices()

    # The metric of the cell should be conserved after applying the operation.
    metric_cc = cell_cv.dot(cell_cv.T)
    metric_scc = np.einsum('sij, jk, slk -> sil',
                           U_scc, metric_cc, U_scc,
                           optimize=True)
    if _backwards_compatible:
        # Wrong units, tolerance is in Å and metric is in lattice basis.
        mask_s = abs(metric_scc - metric_cc).sum(2).sum(1) <= tol
    else:
        L_c = metric_cc.diagonal()**0.5
        tol_cc = np.add.outer(L_c, L_c) * tol
        err_scc = abs(metric_scc - metric_cc)
        mask_s = (err_scc <= tol_cc).all(axis=(1, 2))
    U_scc = U_scc[mask_s]

    # Operation must not swap axes which have different
    # kinds of boundary conditions.
    pbc_cc = np.logical_xor.outer(pbc_c, pbc_c)
    mask_s = ~U_scc[:, pbc_cc].any(axis=1)
    U_scc = U_scc[mask_s]
    return U_scc

def ensure_group(cls,
                 *,
                 cell: ArrayLike1D | ArrayLike2D,
                 rotations: ArrayLike3D | None = None,
                 tolerance: float | None = None,
                 _backwards_compatible=False):
    nsyms = rotations.shape[0]

    M_sscc = np.einsum('sab,pbc->spac', rotations, rotations)
    M_scc = np.unique(M_sscc.reshape((-1, 3, 3)), axis=0)
    if M_scc.shape[0] == nsyms:
        print('log succes, group found, etc.')
        return cls(cell=cell,
                   rotations=rotations,
                   tolerance=tolerance,
                   _backwards_compatible=_backwards_compatible)

    not_a_group = True
    new_nsyms = nsyms
    new_rotations = rotations
    while not_a_group:
        badness = np.zeros((new_nsyms, new_nsyms), dtype=int)
        for s1, M_scc in enumerate(M_sscc):
            for s2, M_cc in enumerate(M_scc):
                contained_operation = any([(M_cc == rotation).all()
                                           for rotation in rotations])
                if not contained_operation:
                    badness[s1, s2] = 1
        np.set_printoptions(threshold=np.inf)

        badness_measure = badness.sum(axis=0) + badness.sum(axis=1)
        if sum(badness_measure) == 0:
            not_a_group = False
            return cls(cell=cell,
                       rotations=new_rotations,
                       tolerance=tolerance,
                       _backwards_compatible=_backwards_compatible)

        bad_operations = np.squeeze(np.argwhere(badness_measure == np.max(badness_measure)))
        print(bad_operations)
        new_rotations = np.delete(new_rotations, bad_operations, axis=0)  # Use boolean masking instead?
        print(new_rotations.shape)
        new_nsyms = new_rotations.shape[0]
        M_sscc = np.einsum('sab,pbc->spac', new_rotations, new_rotations)


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

    print([s[0] for s in symmetries])

    sym = Symmetries(cell=sym.cell_cv,
                     rotations=[s[0] for s in symmetries],
                     translations=[s[1] for s in symmetries],
                     atommaps=[s[2] for s in symmetries],
                     tolerance=sym.tolerance,
                     _backwards_compatible=sym._backwards_compatible)
    if debug:
        sym.check_positions(relpos_ac)

    return sym






# @cache
def generate_all_symmetry_matrices() -> np.ndarray:
    # Symmetry operations as matrices in a basis of lattice vectors.
    # Operation is a 3x3 matrix with possible elements -1, 0, 1, thus
    # there are 3**9 = 19683 possible matrices.

    combinations = 1 - np.indices([3] * 9, dtype=np.int8)
    U_scc = combinations.reshape((3, 3, 3**9)).transpose((2, 0, 1))

    # Matrices must represent rotations with determinant 1
    # or reflections and rotoinversions with determinant -1.

    U_scc = U_scc[abs(leibniz_determinant_3x3(U_scc)) == 1]  # Reduce to 6960 matrices
    return U_scc


def leibniz_determinant_3x3(M: np.ndarray) -> np.ndarray:
    """For calculating the determinant of a collection of 3x3 matrices
    while preserving the dtype of the array since np.linalg.det always
    returns a floating-point number for each matrix"""
    assert M.shape[-2] == 3 and M.shape[-1] == 3
    return (M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2]
                            - M[..., 1, 2] * M[..., 2, 1])
            + M[..., 0, 1] * (M[..., 1, 2] * M[..., 2, 0]
                              - M[..., 1, 0] * M[..., 2, 2])
            + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1]
                              - M[..., 1, 1] * M[..., 2, 0]))
