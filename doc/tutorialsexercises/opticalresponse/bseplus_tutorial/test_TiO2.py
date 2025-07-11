"""Test the calculated 2D polarizability."""

import numpy as np
import pytest
from gpaw.test import findpeak


def refractive_index(Realpart, Impart):
    n = (np.sqrt(np.pi * 4 * (Realpart + Impart * 1j) + 1)).real
    return n


def eels(chi_real, chi_im):
    eps = np.pi * 4 * (chi_real.real + chi_im * 1j) + 1
    return (-eps**(-1)).imag


def test():
    """Test data in n_TiO2.png and eels_TiO2.png figure"""
    # Load data
    chi_bsep = np.load('chi_TiO2_BSEPlus.npy')
    chi_bse = np.load('chi_TiO2_BSE.npy')
    chi_rpa = np.load('chi_TiO2_RPA.npy')
    w = np.linspace(0, 50, 5001)

    n_bsep = refractive_index(-(chi_bsep[:, 0, 0]).real,
                              -(chi_bsep[:, 0, 0]).imag)
    n_bse = refractive_index(-(chi_bse[:, 0, 0]).real,
                             -(chi_bse[:, 0, 0]).imag)
    n_rpa = refractive_index(-(chi_rpa[:, 0, 0]).real,
                             -(chi_rpa[:, 0, 0]).imag)

    eels_bsep = eels(-(chi_bsep[:, 0, 0]).real, -(chi_bsep[:, 0, 0]).imag)
    eels_bse = eels(-(chi_bse[:, 0, 0]).real, -(chi_bse[:, 0, 0]).imag)
    eels_rpa = eels(-(chi_rpa[:, 0, 0]).real, -(chi_rpa[:, 0, 0]).imag)

    # Test static refractive index
    assert n_bsep[0] == pytest.approx(2.43, abs=0.01)
    assert n_bse[0] == pytest.approx(1.97, abs=0.01)
    assert n_rpa[0] == pytest.approx(2.34, abs=0.01)

    # Test high-frequency refractive index
    assert n_bsep[-1] == pytest.approx(0.78, abs=0.01)
    assert n_bse[-1] == pytest.approx(0.99, abs=0.01)
    assert n_rpa[-1] == pytest.approx(0.78, abs=0.01)

    # Test maxima in refractive index
    w_max_bsep, n_max_bsep = findpeak(w, n_bsep)
    w_max_bse, n_max_bse = findpeak(w, n_bse)
    w_max_rpa, n_max_rpa = findpeak(w, n_rpa)

    assert n_max_bsep == pytest.approx(4.20, abs=0.01)
    assert w_max_bsep == pytest.approx(3.92, abs=0.01)

    assert n_max_bse == pytest.approx(3.91, abs=0.01)
    assert w_max_bse == pytest.approx(3.94, abs=0.01)

    assert n_max_rpa == pytest.approx(3.64, abs=0.01)
    assert w_max_rpa == pytest.approx(4.45, abs=0.01)

    # Test maxima in eels
    w_max_bsep, eels_max_bsep = findpeak(w, eels_bsep)
    w_max_bse, eels_max_bse = findpeak(w, eels_bse)
    w_max_rpa, eels_max_rpa = findpeak(w, eels_rpa)

    assert eels_max_bsep == pytest.approx(1.43, abs=0.01)
    assert w_max_bsep == pytest.approx(48.04, abs=0.01)

    assert eels_max_bse == pytest.approx(33.14, abs=0.01)
    assert w_max_bse == pytest.approx(9.67, abs=0.01)

    assert eels_max_rpa == pytest.approx(1.45, abs=0.01)
    assert w_max_rpa == pytest.approx(48.04, abs=0.01)


if __name__ == '__main__':
    test()
