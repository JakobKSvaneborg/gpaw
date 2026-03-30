import pytest
from ase.build import molecule
from gpaw.dft import DFT, PW
from gpaw.new.calculation import DFTCalculation
from gpaw.new.pw.nschse import NonSelfConsistentHybridXCCalculator


def test_exx(gpw_files):
    dft = DFTCalculation.from_gpw_file(gpw_files['n2_pw'])
    exx = NonSelfConsistentHybridXCCalculator.from_dft_calculation(
        dft, 'EXX')
    a, b = exx.calculate(dft.ibzwfs, 0, 8)
    print(a)
    print(b)


def test_2h():
    """This special 2D unit-cell with 14x14 k-points and ecut=200
    triggers a negative value for the Fourier transform of the
    Wigner Seitz truncated Coulomb potential.
    """
    a = 4.18
    k = 14
    ecut = 200.0
    h2 = molecule('H2', cell=[a, a, 18.4, 90, 90, 120], pbc=(1, 1, 0))
    h2.center()
    dft = DFT(h2, mode=PW(ecut, force_complex_dtype=True), kpts=(k, k, 1))
    dft.converge()
    exx = NonSelfConsistentHybridXCCalculator.from_dft_calculation(
        dft, 'EXX')
    elda_skn, eexx_skn = exx.calculate(dft.ibzwfs, ibz_indices=[0])
    assert elda_skn[0, 0] == pytest.approx([-10.12370942, -0.57308455])
    assert eexx_skn[0, 0] == pytest.approx([-16.12300273,  1.30214902])
