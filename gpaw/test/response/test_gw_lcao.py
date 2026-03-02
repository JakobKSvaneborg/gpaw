import pytest
from gpaw.response.g0w0 import G0W0
from gpaw.new.calculation import DFTCalculation


@pytest.mark.ci
@pytest.mark.response
@pytest.mark.old_gpaw_only
def test_lcao_gw(in_tmp_dir, gpw_files, mpi):
    dft = DFTCalculation.from_gpw_file(gpw_files['diamond_lcao'], comm=mpi.comm)
    dft.change_mode('pw')
    dft.write_gpw_file('diamond_pw.gpw', include_wfs=True)

    gw = G0W0(
        'diamond_pw.gpw',
        integrate_gamma='WS',
        ecut=100,
        eta=0.1,
        bands=(0, 8),
        world=mpi.comm)

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

    assert eps_0 == pytest.approx(expected_eps_0, abs=0.01)
    assert f_0 == pytest.approx(expected_f_0, abs=0.01)
    assert Z_0 == pytest.approx(expected_Z_0, abs=0.01)
    assert qp_0 == pytest.approx(expected_qp_0, abs=0.01)

    assert eps[0][0][4] == pytest.approx(17.5784, abs=0.01)
    assert f[0][0][4] == pytest.approx(0.0, abs=0.01)
    assert qp[0][0][4] == pytest.approx(20.370, abs=0.01)
    assert Z[0][0][4] == pytest.approx(1.339, abs=0.01)
