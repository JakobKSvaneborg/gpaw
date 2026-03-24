from gpaw.new.pw.nschse import NonSelfConsistentHybridXCCalculator
from gpaw.new.calculation import DFTCalculation


def test_exx(gpw_files):
    dft = DFTCalculation.from_gpw_file(gpw_files['n2_pw'])
    exx = NonSelfConsistentHybridXCCalculator.from_dft_calculation(
        dft, 'EXX')
    a, b = exx.calculate(dft.ibzwfs, 0, 8)
    print(a)
    print(b)
