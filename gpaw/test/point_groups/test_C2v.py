import pytest

from gpaw import GPAW
from gpaw.point_groups import PointGroup, SymmetryChecker


@pytest.mark.serial
def test_c2v(gpw_files):
    calc = GPAW(gpw_files['h2o_lcao'])
    C = calc.atoms.positions[0]
    pg = PointGroup('C2v')
    sc = SymmetryChecker(pg, C, 2.0)
    symmetries = ''
    for n in range(4):
        print('-' * 70)
        dct = sc.check_band(calc, n)
        sym = dct['symmetry']
        symmetries += sym
        assert dct['characters'][sym] == pytest.approx(1, abs=0.01)
    assert symmetries == 'A1B2A1B1'
