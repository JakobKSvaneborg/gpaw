"""Test CLI-dipole script."""
import pytest

from gpaw.utilities.dipole import main


@pytest.mark.serial
@pytest.mark.parametrize(
    'mode',
    ['pw', 'lcao', 'fd'])
def test_dipole(gpw_files, mode, capsys):
    args = [str(gpw_files[f'h2_{mode}'])]
    main(args)
    out = capsys.readouterr().out
    lines = [line.split() for line in out.splitlines()]
    assert abs(float(lines[6][2])) == pytest.approx(0.55, abs=0.05)
