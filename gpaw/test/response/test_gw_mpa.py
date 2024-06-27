import pytest
import numpy as np
from gpaw.response.g0w0 import G0W0
from ase.units import Hartree as Ha
from gpaw.mpi import world


@pytest.mark.response
def test_mpa_WS(in_tmp_dir, gpw_files, scalapack):
    ref_result = np.array([[[10.989471, 20.627495],
                            [ 5.095711, 15.25836 ],
                            [ 8.465492, 21.623946]]])

    mpa_dict = {'npoles': 4, 'wrange': [0 * Ha, 2 * Ha],
                'varpi': Ha,
                'eta0': 0.01 * Ha,
                'eta_rest': 0.1 * Ha,
                'alpha': 1}

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nblocks=min(2, world.size),
              ecut=60,
              nbands=20,
              integrate_gamma='WS',
              ppa=False,
              mpa=mpa_dict)

    results = gw.calculate()
    print(results)
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)


@pytest.mark.response
def test_mpa(in_tmp_dir, gpw_files, scalapack):
    ref_result = np.asarray([[[11.283458, 21.601906],
                              [5.326717, 16.066114],
                              [8.73869, 22.457025]]])

    mpa_dict = {'npoles': 4, 'wrange': [0 * Ha, 2 * Ha],
                'varpi': Ha,
                'eta0': 0.01 * Ha,
                'eta_rest': 0.1 * Ha,
                'alpha': 1}

    gw = G0W0(gpw_files['bn_pw'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=40,
              ppa=False,
              mpa=mpa_dict)

    results = gw.calculate()
    np.testing.assert_allclose(results['qp'], ref_result, rtol=1e-03)
