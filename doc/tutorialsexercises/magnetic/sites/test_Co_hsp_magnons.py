"""Test the calculated magnon energies."""

import numpy as np
import pytest

from gpaw.response.heisenberg import calculate_fm_magnon_energies


def test():
    # Load data
    magmom = np.load('Co_magmom.npy')
    q_pc = np.load('Co_q_pc.npy')
    rc_r = np.load('Co_rc_r.npy')
    J_pabr = np.load('Co_J_pabr.npy')

    # Calculate the magnon energies
    mm_ar = magmom * np.ones(J_pabr.shape[2:], dtype=float)
    E_pnr = calculate_fm_magnon_energies(J_pabr, q_pc, mm_ar)
    E_pnr = np.sort(E_pnr, axis=1)

    # Test that the magnon energies are constant above a certain cutoff
    assert np.sum(rc_r > 0.9) == 20
    assert np.std(E_pnr[..., rc_r > 0.9], axis=2) == pytest.approx(
        0., abs=0.005)  # σ < 5 meV

    # Test values of the magnon energies at the max cutoff
    assert E_pnr[1:, 0, -1] == pytest.approx([0.45, 0.451, 0.285], abs=0.002)
    assert E_pnr[:, 1, -1] == pytest.approx(
        [0.512, 0.572, 0.451, 0.285], abs=0.002)


if __name__ == '__main__':
    test()
