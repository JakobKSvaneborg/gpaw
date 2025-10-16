import pytest

@pytest.mark.response
def test_lcao_gw(in_tmp_dir):
    from ase.build import bulk
    from gpaw import GPAW
    from gpaw.response.g0w0 import G0W0
    atoms = bulk('C')
    atoms.calc = GPAW(mode='lcao',
                      basis='dzp',
                      kpts={'gamma': True, 'size': (8,8,8)})
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw', mode='all')
    gw = G0W0('gs.gpw',
              nbands=8, integrate_gamma='WS',
              ecut=200,
              eta=0.1, relbands=(0, 8))
    gw.calculate()

