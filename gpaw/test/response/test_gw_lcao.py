import pytest

@pytest.mark.ci
@pytest.mark.response
def test_lcao_gw(in_tmp_dir):
    from ase.build import bulk
    from gpaw import GPAW
    from gpaw.response.g0w0 import G0W0
    
    atoms = bulk('C')
    atoms.calc = GPAW(mode='lcao',
                    basis='dzp',
                    nbands='nao',
                    kpts={'gamma': True, 'size': (1,1,1)})
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw', mode='all')
    gw = G0W0('gs.gpw',
              nbands=((1 + 3) * 2 + 5) * 2, 
              integrate_gamma='WS',
              ecut=100, 
              eta=0.1, bands=(0, 8))
    res = gw.calculate()

    # TODO: Assert for results
    qp = res['qp']
    f = res['f']
    eps = res['eps']
    Z = res['Z']

    eps_0 = eps[0][0][0]
    f_0 = f[0][0][0]

    expected_eps_0 = -8.532558620475898
    expected_f_0 = 1.00000000

    assert eps_0 == pytest.approx(expected_eps_0, abs=0.01)
    assert f_0 == pytest.approx(expected_f_0, abs=0.01)

if __name__ == "__main__":
    test_lcao_gw(None)
