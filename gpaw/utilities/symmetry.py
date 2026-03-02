from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import cache
from itertools import product

import numpy as np
from numpy.typing import NDArray

from gpaw.typing import Array2D, Array3D


def find_set_of_lattice_symmetries(
        cell_cv: Array2D,
        pbc_c: np.ndarray,
        tol: float,
        _backwards_compatible: bool = False) -> Array3D:
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
    U_scc = U_scc[mask_s]
    if not _backwards_compatible:
        # Do not do group check for backwards compatible.
        U_scc = guarantee_lattice_symmetries_form_a_point_group(U_scc)
    return U_scc


@cache
def generate_all_symmetry_matrices() -> NDArray[np.int8]:
    # Symmetry operations as matrices in a basis of lattice vectors.
    # Operation is a 3x3 matrix with possible elements -1, 0, 1, thus
    # there are 3**9 = 19683 possible matrices.

    combinations = 1 - np.indices([3] * 9, dtype=np.int8)
    U_scc = combinations.reshape((3, 3, 3**9)).transpose((2, 0, 1))

    # Matrices must represent rotations with determinant 1
    # or reflections and rotoinversions with determinant -1.

    U_scc = U_scc[abs(leibniz_determinant_3x3(U_scc)) == 1]
    # Reduced to 6960 matrices
    return U_scc


def leibniz_determinant_3x3(M: NDArray) -> NDArray:
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
        initial_U_scc: NDArray[np.int8]) -> NDArray[np.int8]:

    not_a_group = True
    initial_ns = len(initial_U_scc)

    # Cayley table
    M_sscc = np.einsum('sab,zbc->szac', initial_U_scc, initial_U_scc)

    def pop_symmetry(
            U_scc: NDArray[np.int8],
            M_sscc: NDArray[np.int8],
            mask_s: NDArray) -> tuple[NDArray[np.int8], NDArray[np.int8], int]:
        U_scc = U_scc[mask_s]
        M_sscc = M_sscc[mask_s][:, mask_s]
        return U_scc, M_sscc, len(U_scc)

    U_scc = initial_U_scc
    ns = initial_ns
    I3 = np.identity(3, dtype=bool)

    while not_a_group:

        has_inverse_operation = np.zeros(ns, dtype=bool)
        for s, M_scc in enumerate(M_sscc):
            has_inverse_operation[s] = any(
                [(M_cc == I3).all() for M_cc in M_scc])

        if (~has_inverse_operation).any():
            # We can unambigously remove the elements
            # that are not invertible in the set.
            U_scc, M_sscc, ns = pop_symmetry(U_scc, M_sscc,
                                             mask_s=has_inverse_operation)
            continue

        closure_violation = np.zeros((ns, ns), dtype=bool)
        for s1, M_scc in enumerate(M_sscc):
            for s2, M_cc in enumerate(M_scc):
                contained_operation = any([(M_cc == U_cc).all()
                                           for U_cc in U_scc])
                if not contained_operation:
                    closure_violation[s1, s2] = True

        if closure_violation.sum() == 0:
            not_a_group = False
            continue

        easy_closure_violators = np.diag(closure_violation)
        if easy_closure_violators.any():
            # We can unambigously remove the elements that
            # multiply themselves outside of the set.
            U_scc, M_sscc, ns = pop_symmetry(U_scc, M_sscc,
                                             mask_s=~easy_closure_violators)
            continue

        badness_measure = (closure_violation.sum(axis=0)
                           + closure_violation.sum(axis=1))

        worst_elements = np.argwhere(
            badness_measure == np.max(badness_measure))[:, 0]

        if len(worst_elements) == 1:
            # We can safely remove the worst element.
            not_worst_element = np.arange(ns) != worst_elements[0]
            U_scc, M_sscc, ns = pop_symmetry(U_scc, M_sscc,
                                             mask_s=not_worst_element)
            continue

        # Hard mode. It's not clear how to proceed.

        found_two_elements = False
        for s1, s2 in product(worst_elements, worst_elements):
            if s1 == s2:
                continue
            if closure_violation[s1, s2]:
                found_two_elements = True
                two_bad_operations = [s1, s2]
                break

        if found_two_elements:
            # We can either remove symmetry operation
            # two_bad_operations[0] or two_bad_operations[1].
            # In the future, we can calculate how much each operation
            # violates the metric and remove the worst one.
            # For now, we will just randomly choose two_bad_operations[0].
            not_worst_element = np.arange(ns) != two_bad_operations[0]
            U_scc, M_sscc, ns = pop_symmetry(U_scc, M_sscc,
                                             mask_s=not_worst_element)
            continue
        else:
            # Here, we just remove the first entry that is
            # also a "worst element".
            not_worst_element = np.arange(ns) != worst_elements[0]
            U_scc, M_sscc, ns = pop_symmetry(U_scc, M_sscc,
                                             mask_s=not_worst_element)
            continue

    print(f'Log succes, set of {initial_ns} operations reduced '
          f'to point group of {ns} elements, etc.')
    return U_scc


def prune_symmetries(rotation_scc: Array3D,
                     cell_cv: Array2D,
                     relpos_ac: Array2D,
                     id_a: Sequence[int],
                     tol: float,
                     symmorphic: bool = True,
                     _backwards_compatible: bool = False):
    """Remove symmetries that are not satisfied by the atoms."""

    if len(relpos_ac) == 0:
        return rotation_scc, None, None

    # Build lists of atom numbers for each type of atom - one
    # list for each combination of atomic number, setup type,
    # magnetic moment and basis set:
    a_ib = defaultdict(list)
    for a, id in enumerate(id_a):
        a_ib[id].append(a)

    a_b = a_ib[id_a[0]]  # just pick the first species

    def check(rotation_cc, translation_c):
        return check_one_symmetry(rotation_cc, translation_c, cell_cv,
                                  relpos_ac, a_ib,
                                  tol, _backwards_compatible)

    # if supercell disable fractional translations:
    if not symmorphic:
        I3 = np.identity(3, bool)
        ft_bc = relpos_ac[a_b[1:]] - relpos_ac[a_b[0]]
        ft_bc -= np.rint(ft_bc)
        for ft_c in ft_bc:
            a_a = check(I3, ft_c)
            if a_a is not None:
                symmorphic = True
                break

    symmetries = []
    ftsymmetries = []

    # go through all possible symmetry operations
    for rotation_cc in rotation_scc:
        # first ignore fractional translations
        a_a = check(rotation_cc, [0., 0., 0.])
        if a_a is not None:
            symmetries.append((rotation_cc, [0., 0., 0.], a_a))
        elif not symmorphic:
            # check fractional translations
            relposrot_ac = np.dot(relpos_ac, rotation_cc)
            ft_ac = relposrot_ac[a_b] - relpos_ac[a_b[0]]
            ft_ac -= np.rint(ft_ac)
            for ft_c in ft_ac:
                a_a = check(rotation_cc, ft_c)
                if a_a is not None:
                    ftsymmetries.append((rotation_cc, ft_c, a_a))

    # Add symmetry operations with fractional translations at the end:
    symmetries.extend(ftsymmetries)

    rotation_scc = np.array([s[0] for s in symmetries])
    translation_sc = np.array([s[1] for s in symmetries])
    atommap_sa = np.array([s[2] for s in symmetries])

    return rotation_scc, translation_sc, atommap_sa


def check_one_symmetry(rotation_cc, translation_c, cell_cv, relpos_ac, a_ib,
                       tol, _backwards_compatible):
    """Checks whether atoms satisfy one given symmetry operation."""

    a_a = np.zeros(len(relpos_ac), int)
    for a_b in a_ib.values():
        relpos_bc = relpos_ac[a_b]
        for a in a_b:
            relpos_c = np.dot(relpos_ac[a], rotation_cc)
            diff_bc = relpos_c - relpos_bc - translation_c
            diff_bc -= diff_bc.round()
            if _backwards_compatible:
                indices = np.where(abs(diff_bc).max(1) < tol)[0]
            else:
                diff_bv = diff_bc @ cell_cv
                indices = np.where((diff_bv**2).sum(1) < tol**2)[0]
            if len(indices) == 1:
                b = indices[0]
                a_a[a] = a_b[b]
            else:
                assert len(indices) == 0
                return None

    return a_a
