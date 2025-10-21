import pytest

from gpaw.new.ase_interface import GPAW
from gpaw.nlopt.adapters import CollinearGSInfo


@pytest.mark.serial
def test_adapter_pseudo_wfs(gpw_files):
    # Indices
    k = 2
    s = 0
    bands = slice(3, 4)

    calc = GPAW(gpw_files['sic_pw'])

    wfs_fromcalc = calc.dft.ibzwfs._get_wfs(k, s)
    u_G_fromcalc = wfs_fromcalc.psit_nX[bands].data

    gs = CollinearGSInfo(calc)
    wfs_s = gs.ibzwfs.wfs_qs[k]
    wfs = gs.get_wfs(wfs_s, s)
    _, u_G = gs.get_plane_wave_coefficients(wfs, bands=bands, spin=s)

    # Test that adapter outputs expected pseudo-wf coefficients
    assert u_G == pytest.approx(u_G_fromcalc, 1e-10)
