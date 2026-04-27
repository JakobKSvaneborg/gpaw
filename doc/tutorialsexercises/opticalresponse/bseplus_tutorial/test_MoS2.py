"""Test the calculated 2D polarizability."""

import numpy as np
import pytest
from scipy.signal import find_peaks
from gpaw.test import findpeak


class approx:
    def __init__(self, x, abs=1e-6):
        self.x = x
        self.abs = abs

    def __eq__(self, other):
        print(self.x, other, other - self.x, self.abs)
        return abs(self.x - other) < self.abs


pytest.approx = approx


def test():
    """Test data in eels_MoS2.png and eels_MoS2_low_frequencies.png figure"""
    # Load data
    chi_bsep = np.load('chi_MoS2_BSEPlus.npy')
    chi_bse = np.load('chi_MoS2_BSE.npy')
    chi_rpa = np.load('chi_MoS2_RPA.npy')
    w = np.linspace(0, 50, 5001)

    eels_bsep = -chi_bsep[:, 0, 0].imag
    eels_bse = -chi_bse[:, 0, 0].imag
    eels_rpa = -chi_rpa[:, 0, 0].imag

    # Test static limit
    assert eels_bsep[0] == pytest.approx(0)
    assert eels_bse[0] == pytest.approx(0)
    assert eels_rpa[0] == pytest.approx(0)

    # Test maxima
    w_max_bsep, chi_max_bsep = findpeak(w, eels_bsep)
    w_max_bse, chi_max_bse = findpeak(w, eels_bse)
    w_max_rpa, chi_max_rpa = findpeak(w, eels_rpa)

    assert chi_max_bsep == pytest.approx(1.640, abs=0.01)
    assert w_max_bsep == pytest.approx(17.024, abs=0.01)

    assert chi_max_bse == pytest.approx(56.174, abs=1.0)
    assert w_max_bse == pytest.approx(8.482, abs=0.03)

    assert chi_max_rpa == pytest.approx(1.987, abs=0.01)
    assert w_max_rpa == pytest.approx(17.046, abs=0.01)

    # Test the two exciton peaks
    peaks_bsep, props_bsep = find_peaks(eels_bsep, height=0.02)
    peaks_bse, props_bse = find_peaks(eels_bse, height=0.02)
    energies_bsep = w[peaks_bsep]
    heights_bsep = props_bsep['peak_heights']
    energies_bse = w[peaks_bse]
    heights_bse = props_bse['peak_heights']

    # A exciton
    idx_exc1 = 0
    assert energies_bsep[idx_exc1] == pytest.approx(1.89, abs=0.01)
    assert heights_bsep[idx_exc1] == pytest.approx(0.049, abs=0.001)
    assert energies_bse[idx_exc1] == pytest.approx(1.89, abs=0.01)
    assert heights_bse[idx_exc1] == pytest.approx(0.040, abs=0.001)

    # B_exciton
    assert energies_bsep[1] == pytest.approx(2.03, abs=0.01)
    assert heights_bsep[1] == pytest.approx(0.077, abs=0.003)
    assert energies_bse[1] == pytest.approx(2.03, abs=0.01)
    assert heights_bse[1] == pytest.approx(0.062, abs=0.003)


if __name__ == '__main__':
    test()
