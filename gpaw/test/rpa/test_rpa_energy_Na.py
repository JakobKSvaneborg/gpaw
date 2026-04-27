import pytest
from gpaw.xc.rpa import RPACorrelation


@pytest.mark.rpa
@pytest.mark.response
def test_rpa_rpa_energy_Na(in_tmp_dir, gpw_files):
    ecut = 120
    rpa = RPACorrelation(
        gpw_files['na_pw'],
        txt=f'rpa_{ecut}s.txt',
        ecut=[ecut])
    E = rpa.calculate()
    assert E == pytest.approx(-1.106, abs=0.005)
