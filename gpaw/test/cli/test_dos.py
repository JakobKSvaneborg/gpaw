"""Test CLI-dos command."""
import pytest

from gpaw.cli.main import main


@pytest.mark.serial
@pytest.mark.parametrize(
    'options',
    ['',
     '-t',
     '-i',
     '-a Li-sp',
     '-w 0',
     '-r -5 5',
     '-n 50',
     '--soc'])
def test_dos(gpw_files, options):
    args = ('dos ' + str(gpw_files['bcc_li_pw']) + ' ' + options).split()
    main(args)
