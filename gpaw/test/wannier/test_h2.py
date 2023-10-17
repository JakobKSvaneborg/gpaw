import pytest
from gpaw import GPAW
from gpaw.wannier import calculate_overlaps


@pytest.mark.wannier
@pytest.mark.serial
@pytest.mark.parametrize('mode', ['pw', 'fd', 'lcao'])
def test_wan_h2(gpw_files, mode, in_tmp_dir, wannier90):
    calc = GPAW(gpw_files[f'h2_{mode}'])
    overlaps = calculate_overlaps(calc, n1=0, n2=1, nwannier=1,
                                  projections={0: 's'})
    wan = overlaps.localize_er(verbose=True)
    print(wan.centers)
    x = calc.atoms.positions[:, 0].mean()
    assert wan.centers[0, 0] == pytest.approx(x, abs=1e-7)

    wan = overlaps.localize_w90()

    x90 = wan.centers[0, 0] % calc.atoms.cell[0, 0]
    assert x90 == pytest.approx(x, abs=1e-7)
