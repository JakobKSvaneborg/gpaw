import pytest


@pytest.mark.ci
@pytest.mark.response
@pytest.mark.old_gpaw_only
def test_lcao_gw(in_tmp_dir):
    from ase.build import bulk
    from gpaw import GPAW
    from gpaw.response.g0w0 import G0W0

    atoms = bulk('C')
    atoms.calc = GPAW(
        mode="lcao",
        basis="dzp",
        nbands="nao",
        kpts={
            "gamma": True,
            "size": (2, 2, 2), }, )
    atoms.get_potential_energy()
    atoms.calc.write('gs.gpw', mode='all')
    gw = G0W0('gs.gpw',
              integrate_gamma='WS',
              ecut=100,
              eta=0.1, bands=(0, 8))
    res = gw.calculate()

    qp = res['qp']
    f = res['f']
    eps = res['eps']
    Z = res['Z']

    eps_0 = eps[0][0][0]
    f_0 = f[0][0][0]
    qp_0 = qp[0][0][0]
    Z_0 = Z[0][0][0]

    expected_eps_0 = -9.35566765642537
    expected_f_0 = 1.00000000
    expected_qp_0 = -9.96584878
    expected_Z_0 = 0.34982126

    assert eps_0 == pytest.approx(expected_eps_0, abs=0.001)
    assert f_0 == pytest.approx(expected_f_0, abs=0.001)
    assert Z_0 == pytest.approx(expected_Z_0, abs=0.001)
    assert qp_0 == pytest.approx(expected_qp_0, abs=0.001)

    assert eps[0][0][4] == pytest.approx(17.5784, abs=0.001)
    assert f[0][0][4] == pytest.approx(0.0, abs=0.001)
    assert qp[0][0][4] == pytest.approx(21.510, abs=0.001)
    assert Z[0][0][4] == pytest.approx(2.046, abs=0.001)


if __name__ == "__main__":
    test_lcao_gw(None)
