import pytest
import numpy as np
from gpaw.response.df import DielectricFunction
from gpaw.response.qeh import GPAW_ChiCalc
from ase.parallel import world

"""
xxx QEH module seem to require at least 6x6x1 kpoints.
    -this should be investigated
xxx Often fails with unreadable errors in interpolation.
    -arrays should be checked with assertions and readable errors
    should be raised.
xxx isotropic_q = False is temporarily turned off. However,
    most features require isotropic_q = True anyway.
    Should we remove the option or should we expand QEH to handle
    non-isotropic q?
"""


def dielectric(calc, domega, omega2, rate=0.0):
    diel = DielectricFunction(calc=calc,
                              frequencies={'type': 'nonlinear',
                                           'omegamax': 10,
                                           'domega0': domega,
                                           'omega2': omega2},
                              nblocks=1,
                              ecut=10,
                              rate=rate,
                              truncation='2D')
    return diel


@pytest.mark.dielectricfunction
@pytest.mark.serial
@pytest.mark.response
def test_basics(in_tmp_dir, gpw_files):
    pytest.importorskip('qeh')
    from qeh.bb_calculator.bb_calculator import BuildingBlock
    from qeh import Heterostructure, interpolate_building_blocks

    class FragileBB(BuildingBlock):
        def append_chi_2D(self, *args, **kwargs):
            if not hasattr(self, 'doom') and self.last_q_idx == 0:
                self.doom = 0
            self.doom += 1  # Advance doom
            print('doom', self.doom)
            if self.doom == 9:
                raise ValueError('Cthulhu awakens')
            BuildingBlock.append_chi_2D(self, *args, **kwargs)

    df = dielectric(gpw_files['graphene_pw'], 0.2, 0.6, rate=0.001)
    df2 = dielectric(gpw_files['mos2_pw'], 0.1, 0.5)

    # Testing to compute building block
    chicalc = GPAW_ChiCalc(df)
    chicalc2 = GPAW_ChiCalc(df2)
    bb1 = BuildingBlock('graphene', chicalc, q_max=0.6)
    bb2 = BuildingBlock('mos2', chicalc2, q_max=0.6)
    bb1.calculate_building_block()
    bb2.calculate_building_block()

    # Test restart calculation
    bb3 = FragileBB('mos2_rs', chicalc2, q_max=0.6)
    with pytest.raises(ValueError, match='Cthulhu*'):
        bb3.calculate_building_block()
    can_load = bb3.load_chi_file()
    assert can_load
    assert not bb3.complete
    bb3.calculate_building_block()
    can_load = bb3.load_chi_file()
    assert can_load
    assert bb3.complete
    data = np.load('mos2-chi.npz')
    data2 = np.load('mos2_rs-chi.npz')
    assert np.allclose(data['chiM_qw'], data2['chiM_qw'])

    assert np.allclose(data['chiMD_qw'], np.zeros(data['chiMD_qw'].shape))
    assert np.allclose(data['chiDM_qw'], np.zeros(data['chiDM_qw'].shape))

    interpolate_building_blocks(BBfiles=['graphene'], BBmotherfile='mos2')

    # test qeh interface
    HS = Heterostructure(structure=['mos2_int', 'graphene_int'],
                         d=[5],
                         wmax=0,
                         d0=5)
    chi = HS.get_chi_matrix()
    correct_val = 0.018863784898982505 + 0.00019467687211225916j

    assert np.amax(chi) == pytest.approx(correct_val)

    # test equal building blocks
    HS = Heterostructure(structure=['2mos2_int'],
                         d=[5],
                         wmax=0,
                         d0=5)
    chi = HS.get_chi_matrix()

    HS = Heterostructure(structure=['mos2_int', 'mos2_int'],
                         d=[5],
                         wmax=0,
                         d0=5)
    chi_new = HS.get_chi_matrix()
    assert np.allclose(chi, chi_new)
    correct_val = 0.018238064281452426 + 8.081233843428657e-05j
    assert np.amax(chi) == pytest.approx(correct_val)


@pytest.mark.dielectricfunction
@pytest.mark.response
@pytest.mark.serial
def test_off_diagonal_chi(in_tmp_dir, gpw_files):
    pytest.importorskip('qeh')
    from qeh.bb_calculator.bb_calculator import BuildingBlock

    df = dielectric(gpw_files['IBiTe_pw_monolayer'], 0.1, 0.5)
    chicalc = GPAW_ChiCalc(df)
    bb = BuildingBlock('IBiTe', chicalc, q_max=0.6)
    bb.calculate_building_block()
    can_load = bb.load_chi_file()
    assert can_load
    chiDM_qw = bb.chiDM_qw
    chiMD_qw = bb.chiMD_qw

    assert np.allclose(chiDM_qw[7, 4],
                       (2.3932924219802144e-07 + 2.05357322173625e-05j))
    assert np.allclose(chiDM_qw[0, 0],
                       (-5.169878828456423e-24 + 1.1126815599304747e-08j))
    assert np.allclose(chiMD_qw[0, 0],
                       (-8.271806125530277e-25 - 1.1126815599304756e-08j))
    assert np.allclose(chiMD_qw[10, 8],
                       (-0.0013460307110321652 - 0.013654449218892167j))
    assert np.allclose(chiMD_qw[9, 9],
                       (-1.857657321140787e-05 - 5.607966877981631e-05j))

# test limited features that should work in parallel


@pytest.mark.skipif(world.size == 1, reason='Features already tested '
                    'in serial in test_basics')
@pytest.mark.skipif(world.size > 6, reason='Parallelization for '
                    'small test-system broken for many cores')
@pytest.mark.dielectricfunction
@pytest.mark.response
def test_bb_parallel(in_tmp_dir, gpw_files):
    pytest.importorskip('qeh')
    from qeh.bb_calculator.bb_calculator import BuildingBlock

    df = dielectric(gpw_files['mos2_pw'], 0.1, 0.5)
    chicalc = GPAW_ChiCalc(df)
    bb1 = BuildingBlock('mos2', chicalc, q_max=0.6)
    bb1.calculate_building_block()
    # Make sure that calculation is finished before loading data file
    world.barrier()
    data = np.load('mos2-chi.npz')

    maxM = np.amax(abs(data['chiM_qw']))
    assert maxM == pytest.approx(0.25076046494995663)
    maxD = np.amax(abs(data['chiD_qw']))
    assert maxD == pytest.approx(0.8448734155764457)
