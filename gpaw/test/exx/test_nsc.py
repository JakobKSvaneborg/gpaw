from gpaw.new.pw.nschse import NonSelfConsistentHybridXCCalculator
from gpaw.new.calculation import DFTCalculation
from ase.build import molecule
from gpaw.dft import DFT, PW


def test_exx(gpw_files):
    dft = DFTCalculation.from_gpw_file(gpw_files['n2_pw'])
    exx = NonSelfConsistentHybridXCCalculator.from_dft_calculation(
        dft, 'EXX')
    a, b = exx.calculate(dft.ibzwfs, 0, 8)
    print(a)
    print(b)


def test_2h():
    h2 = molecule('H2', cell=[3, 3, 15], pbc=1)
    h2.center()
    dft = DFT(h2, mode=PW(800, force_complex_dtype=True), kpts=(6, 7, 1))
    dft.converge()
    exx = NonSelfConsistentHybridXCCalculator.from_dft_calculation(
        dft, 'EXX')
    a, b = exx.calculate(dft.ibzwfs)
    print(a)
    print(b)


test_2h()
