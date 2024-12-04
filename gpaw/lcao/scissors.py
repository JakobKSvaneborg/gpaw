"""Scissors operator for LCAO."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from ase.units import Ha

from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.new.calculation import DFTCalculation
from gpaw.new.lcao.eigensolver import LCAOEigensolver
from gpaw.new.symmetry import Symmetries


def non_self_consistent_scissors_shift(
        shifts: Sequence[tuple[float, float, int]],
        dft: DFTCalculation) -> np.ndarray:
    """Apply non self-consistent scissors shift.

    Return eigenvalues ase a::

      (nspins, nibzkpts, nbands)

    shaped ndarray in eV units.

    The *shifts* are given as a sequence of tuples
    (energy shifts in eV)::

        [(<shift for occupied states>,
          <shift for unoccupied states>,
          <number of atoms>),
         ...]

    Here we open a gap for states on atoms with indices 3, 4 and 5::

      eig_skM = non_self_consistent_scissors_shift(
          [(0.0, 0.0, 3),
           (-0.5, 0.5, 3)],
          dft)
    """
    ibzwfs = dft.ibzwfs
    check_symmetries(ibzwfs.ibz.symmetries, shifts)
    shifts = [(homo / Ha, lumo / Ha, natoms)
              for homo, lumo, natoms in shifts]
    matcalc = dft.scf_loop.hamiltonian.create_hamiltonian_matrix_calculator(
        dft.potential)
    matcalc = MyMatCalc(matcalc, shifts)
    eig_skn = np.zeros((ibzwfs.nspins, len(ibzwfs.ibz), ibzwfs.nbands))
    for wfs in ibzwfs:
        H_MM = matcalc.calculate_matrix(wfs)
        eig_M = H_MM.eighg(wfs.L_MM, wfs.domain_comm)
        eig_skn[wfs.spin, wfs.k] = eig_M
    ibzwfs.kpt_comm.sum(eig_skn)
    return eig_skn * Ha


def check_symmetries(symmetries: Symmetries,
                     shifts: Sequence[tuple[float, float, int]]) -> None:
    """Make sure shifts don't break any symmetries.

    >>> from gpaw.new.symmetry import create_symmetries_object
    >>> from ase import Atoms
    >>> atoms = Atoms('HH', [(0, 0, 1), (0, 0, -1)], cell=[3, 3, 3])
    >>> sym = create_symmetries_object(atoms)
    >>> check_symmetries(sym, [(1.0, 1.0, 1)])
    Traceback (most recent call last):
        ...
    ValueError: A symmetry maps atom 0 onto atom 1,
    but those atoms have different scissors shifts
    """
    b_sa = symmetries.atommap_sa
    shift_a = []
    for ho, lu, natoms in shifts:
        shift_a += [(ho, lu)] * natoms
    shift_a += [(0.0, 0.0)] * (b_sa.shape[1] - len(shift_a))
    for b_a in b_sa:
        for a, b in enumerate(b_a):
            if shift_a[a] != shift_a[b]:
                raise ValueError(f'A symmetry maps atom {a} onto atom {b},\n'
                                 'but those atoms have different '
                                 'scissors shifts')


class ScissorsLCAOEigensolver(LCAOEigensolver):
    def __init__(self,
                 basis,
                 shifts: Sequence[tuple[float, float, int]],
                 symmetries: Symmetries):
        """Scissors-operator eigensolver."""
        check_symmetries(symmetries, shifts)
        super().__init__(basis)
        self.shifts = []
        for homo, lumo, natoms in shifts:
            self.shifts.append((homo / Ha, lumo / Ha, natoms))

    def iterate1(self, wfs, matrix_calculator):
        super().iterate1(wfs, MyMatCalc(matrix_calculator, self.shifts))

    def __repr__(self):
        txt = DirectLCAO.__repr__(self)
        txt += '\n    Scissors operators:\n'
        a1 = 0
        for homo, lumo, natoms in self.shifts:
            a2 = a1 + natoms
            txt += (f'      Atoms {a1}-{a2 - 1}: '
                    f'VB: {homo * Ha:+.3f} eV, '
                    f'CB: {lumo * Ha:+.3f} eV\n')
            a1 = a2
        return txt


class MyMatCalc:
    def __init__(self, matcalc, shifts):
        self.matcalc = matcalc
        self.shifts = shifts

    def calculate_matrix(self, wfs):
        H_MM = self.matcalc.calculate_matrix(wfs)

        try:
            nocc = int(round(wfs.occ_n.sum()))
        except ValueError:
            return H_MM

        C_nM = wfs.C_nM.data
        S_MM = wfs.S_MM.data
        # assert abs(S_MM - S_MM.T.conj()).max() < 1e-10

        # Find Z=S^(1/2):
        e_N, U_MN = np.linalg.eigh(S_MM)
        # We now have: S_MM @ U_MN = U_MN @ diag(e_N)
        Z_MM = U_MN @ (e_N[np.newaxis]**0.5 * U_MN).T.conj()

        # Density matrix:
        A_nM = C_nM[:nocc].conj() @ Z_MM
        R_MM = A_nM.conj().T @ A_nM

        M1 = 0
        a1 = 0
        for homo, lumo, natoms in self.shifts:
            a2 = a1 + natoms
            M2 = M1 + sum(setup.nao for setup in wfs.setups[a1:a2])
            l_n, V_mn = np.linalg.eigh(R_MM[M1:M2, M1:M2])
            V_Mn = Z_MM[:, M1:M2] @ V_mn
            L_1n = (homo - lumo) * l_n[np.newaxis] + lumo
            H_MM.data += V_Mn @ (L_1n * V_Mn).T.conj()
            a1 = a2
            M1 = M2

        return H_MM
