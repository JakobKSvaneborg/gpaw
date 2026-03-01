from __future__ import annotations

import numpy as np
from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from gpaw import debug
from gpaw.new import zips
from gpaw.typing import Array2D, Array3D

if TYPE_CHECKING:
    from gpaw.new.symmetry import Symmetries


def find_set_of_lattice_symmetries(cell_cv: Array2D,
                                   pbc_c: tuple,
                                   tol,
                                   _backwards_compatible=False) -> Array3D:
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
    returns a floating-point number for each matrix."""
    assert M.shape[-2] == 3 and M.shape[-1] == 3
    return (M[..., 0, 0] * (M[..., 1, 1] * M[..., 2, 2]
                            - M[..., 1, 2] * M[..., 2, 1])
            + M[..., 0, 1] * (M[..., 1, 2] * M[..., 2, 0]
                              - M[..., 1, 0] * M[..., 2, 2])
            + M[..., 0, 2] * (M[..., 1, 0] * M[..., 2, 1]
                              - M[..., 1, 1] * M[..., 2, 0]))


def guarantee_lattice_symmetries_form_a_point_group(
    initial_rotation_scc: Array3D) -> Array3D:

    not_a_group = True
    initial_ns = initial_rotation_scc.shape[0]

    # Cayley table
    M_sscc = np.einsum('sab,zbc->szac',
        initial_rotation_scc, initial_rotation_scc)

    rotation_scc = initial_rotation_scc
    ns = initial_ns
    I3 = np.eye(3, dtype=bool)
    
    while not_a_group:

        has_inverse_operation = np.zeros(ns, dtype=bool)
        for s, (rotation_cc, M_scc) in enumerate(zips(rotation_scc, M_sscc)):
            has_inverse_operation[s] = any(
                [(M_cc == I3).all() for M_cc in M_scc])

        if (~has_inverse_operation).any():
            # We can unambigously remove the elements
            # that are not invertible in the set.
            rotation_scc = rotation_scc[has_inverse_operation]
            ns = rotation_scc.shape[0]
            M_sscc = M_sscc[has_inverse_operation][:, has_inverse_operation]
            continue

        closure_violation = np.zeros((ns, ns), dtype=bool)
        for s1, M_scc in enumerate(M_sscc):
            for s2, M_cc in enumerate(M_scc):
                contained_operation = any([(M_cc == rotation_cc).all()
                                           for rotation_cc in rotation_scc])
                if not contained_operation:
                    closure_violation[s1, s2] = True

        if closure_violation.sum() == 0:
            not_a_group = False
        else:
            easy_closure_violators = np.diag(closure_violation)
            if easy_closure_violators.any():
                # We can unambigously remove the elements that
                # multiply themselves outside of the set.
                rotation_scc = rotation_scc[~easy_closure_violators]
                ns = rotation_scc.shape[0]
                M_sscc = M_sscc[~easy_closure_violators][:, ~easy_closure_violators]
                continue

            badness_measure = (closure_violation.sum(axis=0)
                               + closure_violation.sum(axis=1))

            worst_elements = np.argwhere(
                badness_measure == np.max(badness_measure))[:, 0]

            if len(worst_elements) == 1:
                # We can safely remove the worst element.
                not_worst_element = np.arange(ns) != worst_elements[0]
                rotation_scc = rotation_scc[not_worst_element]
                ns = rotation_scc.shape[0]
                M_sscc = M_sscc[not_worst_element][:, not_worst_element]
                continue

            # Hard mode. It's not clear how to proceed.

            found_two_elements = False
            for s1 in range(ns):
                if s1 not in worst_elements:
                    continue
                for s2 in chain(range(0, s1), range(s1 + 1, ns)):
                    if s2 not in worst_elements:
                        continue
                    if closure_violation[s1, s2] == 1:
                        found_two_elements = True
                        two_bad_operations = [s1, s2]
                        break
                if found_two_elements == True:
                    break

            # We can either remove symmetry operation two_bad_operations[0]
            # or two_bad_operations[1]. In the future, we can calculate how
            # much each operation violates the metric and remove the worst one.
            # For now, we will just randomly choose two_bad_operations[0].

            not_worst_element = np.arange(ns) != two_bad_operations[0]
            rotation_scc = rotation_scc[not_worst_element]
            ns = rotation_scc.shape[0]
            M_sscc = M_sscc[not_worst_element][:, not_worst_element]
            continue

    print(f'Log succes, set of {initial_ns} operations reduced '
          f'to point group of {ns} elements, etc.')
    return rotation_scc.copy()


def prune_symmetries(rotation_scc: Array3D,
                     relpos_ac: Array2D,
                     cell_cv,
                     id_a: Sequence[int],
                     tol: float,
                     symmorphic: bool = True,
                     _backwards_compatible: bool = False):
    """Remove symmetries that are not satisfied by the atoms."""

    if len(relpos_ac) == 0:
        return rotation_scc, 

    # Build lists of atom numbers for each type of atom - one
    # list for each combination of atomic number, setup type,
    # magnetic moment and basis set:
    a_ij = defaultdict(list)
    for a, id in enumerate(id_a):
        a_ij[id].append(a)

    a_j = a_ij[id_a[0]]  # just pick the first species

    def check(op_cc, ft_c):
        return check_one_symmetry(relpos_ac, op_cc, ft_c, a_ij)

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

    return sym


def check_one_symmetry(cell_cv, spos_ac, op_cc, ft_c, a_ia, tolerance, _backwards_compatible):
    """Checks whether atoms satisfy one given symmetry operation."""

    a_a = np.zeros(len(spos_ac), int)
    for b_a in a_ia.values():
        spos_jc = spos_ac[b_a]
        for b in b_a:
            spos_c = np.dot(spos_ac[b], op_cc)
            sdiff_jc = spos_c - spos_jc - ft_c
            sdiff_jc -= sdiff_jc.round()
            if _backwards_compatible:
                indices = np.where(
                    abs(sdiff_jc).max(1) < tolerance)[0]
            else:
                sdiff_jv = sdiff_jc @ cell_cv
                indices = np.where(
                    (sdiff_jv**2).sum(1) < tolerance**2)[0]
            if len(indices) == 1:
                a = indices[0]
                a_a[b] = b_a[a]
            else:
                assert len(indices) == 0
                return None

    return a_a
