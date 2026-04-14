# import pytest

from gpaw import GPAW
from gpaw.hybrids.energy import non_self_consistent_energy as nsc_energy
from gpaw.new.calculation import DFTCalculation
from gpaw.new.pw.hybrids import non_self_consistent_hybrid_xc_energy


def test_h(gpw_files):
    h_calc = GPAW(gpw_files['h_pw'])
    e = nsc_energy(h_calc, 'EXX')
    print(e)


def test_h_new(gpw_files):
    h_dft = DFTCalculation.from_gpw_file(gpw_files['h_pw'])
    e = non_self_consistent_hybrid_xc_energy(h_dft, 'EXX')
    print(e)


if __name__ == '__main__':
    import sys
    test_h_new({'h_pw': sys.argv[1]})
