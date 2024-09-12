import pytest
from gpaw.response.df import DielectricFunction
from ase.parallel import world


def dielectric(calc, domega, omega2, rate=0.0, ecut=10, nblocks=1):
    diel = DielectricFunction(calc=calc,
                              frequencies={'type': 'nonlinear',
                                           'omegamax': 10,
                                           'domega0': domega,
                                           'omega2': omega2},
                              nblocks=nblocks,
                              ecut=ecut,
                              rate=rate,
                              truncation='2D')
    return diel


@pytest.mark.dielectricfunction
@pytest.mark.serial
@pytest.mark.response
def test_basics(in_tmp_dir, gpw_files):
    pytest.importorskip('qeh')
    from gpaw.response.qeh import QEHChiCalc

    df = dielectric(gpw_files['graphene_pw'], 0.1, 0.5, rate=0.01)

    chicalc = QEHChiCalc(df)

    assert len(chicalc.get_q_grid(q_max=0.6)) == 3
    assert len(chicalc.get_q_grid(q_max=2.6)) == 6
    assert len(chicalc.get_q_grid(q_max=2.6)[0].P_rv) == 2
    assert len(chicalc.get_z_grid()) == 30

    q_q = chicalc.get_q_grid(q_max=0.6)
    chi_wGG, G_Gv, wblocks = chicalc.get_chi_wGG(qpoint=q_q[2])

    assert chi_wGG[0, 0, 0] == pytest.approx(-3.134762463291029e-10
                                             + 3.407232927207498e-27j)
    assert chi_wGG[3, 2, 1] == pytest.approx(-2.69008628970302e-10
                                             - 6.74306768078481e-11j)
    assert chi_wGG.shape[1] == len(G_Gv)


@pytest.mark.skipif(world.size == 1, reason='Features already tested '
                    'in serial in test_basics')
@pytest.mark.skipif(world.size > 6, reason='Parallelization for '
                    'small test-system broken for many cores')
@pytest.mark.dielectricfunction
@pytest.mark.response
def test_qeh_parallel(in_tmp_dir, gpw_files):
    pytest.importorskip('qeh')
    from gpaw.response.qeh import QEHChiCalc

    df = dielectric(gpw_files['mos2_pw'], 0.05, 0.5, nblocks=world.size)
    chicalc = QEHChiCalc(df)

    q_q = chicalc.get_q_grid(q_max=0.6)
    chi_wGG, G_Gv, wblocks = chicalc.get_chi_wGG(qpoint=q_q[2])
    chi_wGG = wblocks.all_gather(chi_wGG)
    if world.rank == 0:
        assert chi_wGG.shape[0] == 23
        assert chi_wGG[0, 0, 0] == pytest.approx(-0.0050287263466402875
                                                 + 3.49049213870125e-20j)
        assert chi_wGG[3, 2, 1] == pytest.approx(0.004918844473000315
                                                 + 0.0002505019241197282j)
        assert chi_wGG.shape[1] == len(G_Gv)
