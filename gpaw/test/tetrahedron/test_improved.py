"""Test ITM with half-filled 1-d cosine band."""
from math import pi
import pytest
import numpy as np
from gpaw.tetrahedron import TetrahedronMethod
from gpaw.dos import DOSCalculator


def f(N: int, i: bool):
    rcell = np.diag([1, 0.1, 0.1])
    k = (np.linspace(-0.5, 0.5, N, endpoint=False) + 0.5 / N) * 2 * pi
    e = -np.cos(k)[:, np.newaxis]
    f = np.empty((N, 1))
    w = np.zeros(N) + 1 / N
    spins = np.zeros(N, int)
    t = TetrahedronMethod(rcell, (N, 1, 1), improved=i)
    ef, _ = t._calculate(0.5, e, w, spins, f)
    wfs = WFS(e, ef)
    dos = DOSCalculator(wfs, cell=np.linalg.inv(rcell))
    dosef = dos.raw_dos([ef], spin=0, width=0.0)[0]
    return f.sum() / N, (f * e).sum() / N, ef, dosef


class WFS:
    def __init__(self, e, ef):
        self.eig_skn = e[np.newaxis]
        self.fermi_level = ef
        self.size = (len(e), 1, 1)
        self.bz2ibz_map = None

    def weights(self):
        n = self.size[0]
        return np.zeros(n) + 1 / n

    def eigenvalues(self):
        return self.eig_skn


@pytest.mark.serial
def test_tm_1d():
    # Improved:
    nelectrons, eband, efermi, dosefermi = f(100, True)
    assert nelectrons == pytest.approx(0.5, abs=1e-9)
    assert eband == pytest.approx(-1 / pi, abs=1e-5)
    assert efermi == pytest.approx(0.0, abs=1e-9)
    assert dosefermi == pytest.approx(1 / pi, abs=1e-4)

    # Standard:
    nelectrons, eband100, efermi, dosefermi = f(100, not True)
    assert nelectrons == pytest.approx(0.5, abs=1e-9)
    assert efermi == pytest.approx(0.0, abs=1e-9)
    assert dosefermi == pytest.approx(1 / pi, abs=1e-4)

    # ebands converges as 1/N^2.
    # Extrapolate to infinite number of k-points:
    eband80 = f(80, not True)[1]
    eband = np.polyval(np.polyfit([100**-2, 80**-2],
                                  [eband100, eband80], 1), 0.0)
    assert eband == pytest.approx(-1 / pi, abs=1e-7)
