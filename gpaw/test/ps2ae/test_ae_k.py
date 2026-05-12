import pytest

from gpaw.mpi import world
from gpaw.new.ase_interface import GPAW as GPAW2


@pytest.mark.parametrize('name, tol',
                         [('bcc_li_pw', 3e-5),
                          ('bcc_li_fd', 4e-4)])
def test_ae_k(gpw_files, name, tol):
    """Test normalization of non gamma-point wave functions."""
    if world.size > 1:
        return
    calc = GPAW2(gpw_files[name])
    ae = calc.dft.ibzwfs.get_all_electron_wave_function(
        0, kpt=1, grid_spacing=0.1)
    assert ae.norm2() == pytest.approx(1.0, abs=tol)
