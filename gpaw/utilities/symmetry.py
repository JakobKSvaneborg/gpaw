from __future__ import annotations

import numpy as np
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from gpaw import debug
from gpaw.typing import Array2D

if TYPE_CHECKING:
    from gpaw.new.symmetry import Symmetries

def find_set_of_lattice_symmetries(cell_cv: Arra,
                                   pbc_c: tuple,
                                   tol,
                                   _backwards_compatible=False):
    """Determine set of fixed-point symmetry
    operations compliant with a given lattice."""
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
    return U_scc[mask_s].copy()


def guarantee_lattice_symmetries_form_a_point_group(rotation_scc: ArrayLike3D):
    ns_initial = rotation_scc.shape[0]

    # Cayley table
    M_sscc = np.einsum('sab,zbc->szac', rotation_scc, rotation_scc)

    has_inverse_operation = np.zeros(ns_initial, dtype=bool)
    I3 = np.eye(3, dtype=bool)
    for s, rotation_cc in enumerate(rotation_scc):
        has_inverse_operation[s] = any(
            [(M_cc == I3).all() for M_cc in M_sscc[s]])

    rotation_scc = rotation_scc[has_inverse_operation]
    ns = rotation_scc.shape[0]
    M_sscc = M_sscc[has_inverse_operation][:, has_inverse_operation]
    M_scc = np.unique(M_sscc.reshape((-1, 3, 3)), axis=0)

    print([ns_initial, ns])
    cc
    if M_scc.shape[0] == ns:
        print('log succes, group found, etc.')
        return rotation_scc

    not_a_group = True

    new_ns = ns
    new_rotation_scc = rotation_scc
    while not_a_group:
        badness = np.zeros((new_ns, new_ns), dtype=bool)
        for s1, M_scc in enumerate(M_sscc):
            for s2, M_cc in enumerate(M_scc):
                contained_operation = any([(M_cc == rotation_cc).all()
                                           for rotation_cc in rotation_scc])
                if not contained_operation:
                    badness[s1, s2] = 1
   
        print(badness.astype(int))

        zzz = np.ones(new_rotation_scc.shape[0], dtype=bool)
        zzz[[2]] = 0

        badness_measure_1 = np.zeros((new_ns - 1), dtype=int)
        for s1, M_scc in enumerate(M_sscc[zzz][:, zzz]):
            for s2, M_cc in enumerate(M_scc):
                contained_operation = any([(M_cc == rotation_cc).all()
                                           for rotation_cc in rotation_scc])
                if not contained_operation:
                    badness_measure_1[s1] += 1
                    badness_measure_1[s2] += 1

        print([badness_measure_1, sum(badness_measure_1)])

        zzz[[2]] = 1
        zzz[[8]] = 0

        badness_measure_2 = np.zeros((new_ns - 1), dtype=int)
        for s1, M_scc in enumerate(M_sscc[zzz][:, zzz]):
            for s2, M_cc in enumerate(M_scc):
                contained_operation = any([(M_cc == rotation_cc).all()
                                           for rotation_cc in rotation_scc])
                if not contained_operation:
                    badness_measure_2[s1] += 1
                    badness_measure_2[s2] += 1

        print([badness_measure_2, sum(badness_measure_2)])
        cc

        print(badness.astype(int))
        print(badness[zzz][:, zzz].astype(int))
        badness_measure = badness.sum(axis=0) + badness.sum(axis=1)
        print(badness_measure)
        print(type(badness_measure))
        cc
        if sum(badness_measure) == 0:
            print('log succes, set of {initial ns} operations reduced '
                  'to group of {new_ns} elements, etc.')
            not_a_group = False
            return new_rotations

        bad_operations = np.squeeze(
            np.argwhere(badness_measure == np.max(badness_measure)))
        new_rotation_scc = new_rotation_scc[~bad_operations]
        new_ns = new_rotation_scc.shape[0]
        M_sscc = M_sscc[~bad_operations][:, ~bad_operations]
        M_sscc = np.einsum('sab,pbc->spac', new_rotation_scc, new_rotation_scc)


        print(new_ns)
        cc


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

    from gpaw.new.symmetry import Symmetries
    sym = Symmetries(cell=sym.cell_cv,
                     rotations=[s[0] for s in symmetries],
                     translations=[s[1] for s in symmetries],
                     atommaps=[s[2] for s in symmetries],
                     tolerance=sym.tolerance,
                     _backwards_compatible=sym._backwards_compatible)
    if debug:
        sym.check_positions(relpos_ac)

    return sym


@cache
def generate_all_symmetry_matrices() -> np.ndarray:
    # Symmetry operations as matrices in a basis of lattice vectors.
    # Operation is a 3x3 matrix with possible elements -1, 0, 1, thus
    # there are 3**9 = 19683 possible matrices.

    combinations = 1 - np.indices([3] * 9, dtype=np.int8)
    U_scc = combinations.reshape((3, 3, 3**9)).transpose((2, 0, 1))

    # Matrices must represent rotations with determinant 1
    # or reflections and rotoinversions with determinant -1.

    U_scc = U_scc[abs(leibniz_determinant_3x3(U_scc)) == 1]  # Reduce to 6960 matrices
    return U_scc.copy()


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
