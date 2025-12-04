import pytest

from gpaw.elf import elf_from_dft_calculation
from gpaw.new.ase_interface import GPAW


@pytest.mark.parametrize('name', ['h2_fd', 'bcc_li_pw'])
def test_elf(gpw_files, name):
    dft = GPAW(gpw_files[name]).dft
    e_R = elf_from_dft_calculation(dft)
    print(e_R.data.min())
    print(e_R.data.max())
