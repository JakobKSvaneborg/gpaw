import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from ase.units import Hartree as Ha


@pytest.mark.response
def test_ff(in_tmp_dir, gpw_files, scalapack):
    ref_result = np.asarray([[[11.290542, 21.613643],
                              [5.356609, 16.065227],
                              [8.751158, 23.156368]]])

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              evaluate_sigma=np.linspace(-1, 1, 10),
              ecut=40)

    results = gw.calculate()
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)


@pytest.mark.response
@pytest.mark.parametrize("mpa", [True, False])
def test_ppa(in_tmp_dir, gpw_files, scalapack, mpa):
    ref_result = {False: np.asarray([[[11.30094393, 21.62842077],
                                      [5.33751513, 16.06905725],
                                      [8.75269938, 22.46579489]]]),
                  True: np.asarray([[[11.303942, 21.624428],
                                     [5.346694, 16.06346],
                                     [8.7589, 22.461506]]])}[mpa]
    mpa_dict = {'npoles': 1, 'wrange': [0, 0],
                'varpi': Ha, 'eta0': 1e-10, 'eta_rest': 10000 * Ha,
                'alpha': 1, 'asd':0}

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=40,
              ppa=not mpa,
              mpa=mpa_dict if mpa else False)

    results = gw.calculate()
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)
