import pytest

from gpaw import GPAW
from gpaw.elf import elf_from_dft


@pytest.mark.parametrize(name, ['h2_fd', 'bcc_li_pw'])
def test_elf(gpw_files, name):
    dft = GPAW(gpw_files[name]).dft
    elf_from_dft(dft)
