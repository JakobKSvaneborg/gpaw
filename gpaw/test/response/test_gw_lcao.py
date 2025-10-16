import pytest

@pytest.mark.response
def test_lcao_gw(in_tmp_dir):
    from ase.build import bulk
    from gpaw import GPAW
    from gpaw.response.g0w0 import G0W0
    if 1:
        atoms = bulk('C')
        atoms.calc = GPAW(mode='lcao',
                          basis='dzp',
                          nbands='nao',
                          kpts={'gamma': True, 'size': (8,8,8)})
        atoms.get_potential_energy()
        atoms.calc.write('gs.gpw', mode='all')
    gw = G0W0('gs.gpw',
              nbands=((1 + 3) * 2 + 5) * 2, 
              integrate_gamma='WS',
              ecut=200, 
              eta=0.1, bands=(0, 8))
    gw.calculate()

if __name__ == "__main__":
    test_lcao_gw(None)
