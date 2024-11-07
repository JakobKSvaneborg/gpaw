"""Scissors operator for LCAO."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from ase.units import Ha

from gpaw.lcao.eigensolver import DirectLCAO
from gpaw.new.calculation import DFTCalculation
from gpaw.new.lcao.eigensolver import LCAOEigensolver


def non_self_consistent_scissors_shift(
        shifts: Sequence[tuple[float, float, int]],
        dft: DFTCalculation) -> np.ndarray:
    """Apply non self-consistent scissors shift.

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
    shifts = [(homo / Ha, lumo / Ha, natoms)
              for homo, lumo, natoms in shifts]
    matcalc = dft.scf_loop.hamiltonian.create_hamiltonian_matrix_calculator(
        dft.potential)
    matcalc = MyMatCalc(matcalc, shifts)
    eig_uM = []
    for wfs in dft.ibzwfs:
        H_MM = matcalc.calculate_matrix(wfs)
        eig_M = H_MM.eighg(wfs.L_MM, wfs.domain_comm)
        eig_uM.append(eig_M)
    shape = (dft.ibzwfs.nspins, -1, len(eig_M))
    return np.array(eig_uM).reshape(shape) * Ha


class ScissorsLCAOEigensolver(LCAOEigensolver):
    def __init__(self,
                 basis,
                 shifts: Sequence[tuple[float, float, int]]):
        """Scissors-operator eigensolver."""
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

        S_MM = wfs.S_MM.data
        assert abs(S_MM - S_MM.T.conj()).max() < 1e-10

        C_nM = wfs.C_nM.data
        M1 = 0
        a1 = 0
        for homo, lumo, natoms in self.shifts:
            a2 = a1 + natoms
            M2 = M1 + sum(setup.nao for setup in wfs.setups[a1:a2])
            A_nM = C_nM[:nocc, M1:M2].conj() @ S_MM[M1:M2]
            H_MM.data += homo * A_nM.T.conj() @ A_nM
            A_nM = C_nM[nocc:, M1:M2].conj() @ S_MM[M1:M2]
            H_MM.data += lumo * A_nM.T.conj() @ A_nM
            a1 = a2
            M1 = M2
        return H_MM
