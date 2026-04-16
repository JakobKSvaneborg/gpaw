"""Test gpaw symmetry command."""
import pytest

from gpaw.cli.main import main

result = """\
Number of symmetries: 48

BZ-sampling:
  Number of BZ points: 512
  Number of IBZ points: 29

  Monkhorst-Pack size: [8, 8, 8]
  Monkhorst-Pack shift: [0.0625, 0.0625, 0.0625]

"""


@pytest.mark.serial
def test_symmetry(gpw_files, capsys):
    args = ['symmetry',
            str(gpw_files['bcc_li_pw']),
            '-k',
            '{density:3,gamma:1}']
    main(args)
    out = capsys.readouterr().out
    print(out)
    assert out == result
