import pytest
from gpaw.response.g0w0 import G0W0
import numpy as np
from gpaw.mpi import world


@pytest.mark.response
def test_gw_anisotropic(in_tmp_dir, gpw_files, needs_ase_master, gpaw_new):
    if gpaw_new and world.size > 1:
        pytest.skip('Hybrids not working in parallel with GPAW_NEW=1')
    print(gpw_files)
    gw = G0W0(gpw_files['p4_pw'],
              'gw-test',
              nbands=15,
              ecut=30,
              eta=0.2,
              frequencies={'type': 'nonlinear', 'domega0': 0.3},
              truncation='2D',
              kpts=[(0.5, 0, 0), (0, 0, 0), (0, 0.5, 0)],
              bands=(9, 11))

    e_qp = gw.calculate()['qp']
    assert np.allclose(e_qp, [[[3.80581804, 12.04467414],
                               [3.21438974, 11.23852882],
                               [2.32178208, 11.6378957]]], atol=0.001)
