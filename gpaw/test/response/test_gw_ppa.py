import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from gpaw.mpi import world


@pytest.mark.response
def test_ppa(in_tmp_dir, gpw_files, scalapack):
    ref_result = np.asarray([[[11.30094393, 21.62842077],
                              [5.33751513, 16.06905725],
                              [8.75269938, 22.46579489]]])
    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=40,
              ppa=True)

    results = gw.calculate()
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)


if __name__ == '__main__':
    import sys
    test_ppa(1, {'bn_pw': sys.argv[1]}, 1)
