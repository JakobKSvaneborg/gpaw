import pytest
import numpy as np
from gpaw.response.df import DielectricFunction
from gpaw.response.qeh import BuildingBlock, check_building_blocks
from gpaw.mpi import world, size

"""
xxx QEH module seem to require at least 6x6x1 kpoints.
    -this should be investigated
xxx Often fails with unreadable errors in interpolation.
    -arrays should be checked with assertions and readable errors
xxx add_intraband fails with NotImplementedError in dielctric
    function. -Implement or remove option????
xxx isotropic_q = False is temporarily turned off. However,
    most features require isotropic_q = True anyway.
    Should we remove the option or should we expand QEH to handle
    non-isotropic q?
"""


class FragileBB(BuildingBlock):
    def update_building_block(self, *args, **kwargs):
        if not hasattr(self, 'doom') and self.last_q_idx == 0:
            self.doom = 0
        self.doom += 1  # Advance doom
        print('doom', self.doom)
        if self.doom == 9:
            raise ValueError('Cthulhu awakens')
        BuildingBlock.update_building_block(self, *args, **kwargs)


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


@pytest.mark.serial
@pytest.mark.response
def test_basics(in_tmp_dir, gpw_files):
    qeh = pytest.importorskip('qeh')
    interpolate_building_blocks = qeh.interpolate_building_blocks
    Heterostructure = qeh.Heterostructure

    df = dielectric(gpw_files['graphene_pw'], 0.2, 0.6, rate=0.001)
    df2 = dielectric(gpw_files['mos2_pw'], 0.1, 0.5)

    # Testing to compute building block
    bb1 = BuildingBlock('graphene', df)
    bb2 = BuildingBlock('mos2', df2)
    bb1.calculate_building_block()
    bb2.calculate_building_block()

    # Test restart calculation
    bb3 = FragileBB('mos2_rs', df2)
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

    # Test building blocks are on different grids
    are_equal = check_building_blocks(['mos2', 'graphene'])
    assert not are_equal

    # testing to interpolate
    interpolate_building_blocks(BBfiles=['graphene'], BBmotherfile='mos2')
    are_equal = check_building_blocks(['mos2_int', 'graphene_int'])
    assert are_equal

    # test qeh interface
    HS = Heterostructure(structure=['mos2_int', 'graphene_int'],
                         d=[5],
                         wmax=0,
                         d0=5)
    chi = HS.get_chi_matrix()
    correct_val = 0.018928388759896875 - 0.00018260820184429004j
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
    correct_val = 0.018238059045975367 + 8.08142659593134e-05j
    assert np.amax(chi) == pytest.approx(correct_val)

    # test to interpolate to grid and actual numbers
    q_grid = np.array([0, 0.1])
    w_grid = np.array([0, 0.1])
    bb2.interpolate_to_grid(q_grid=q_grid, w_grid=w_grid)
    data = np.load('mos2_int-chi.npz')

    assert np.allclose(data['omega_w'], np.array([0., 0.00367493]))

    monopole = np.array([[-6.19649236e-10 + 8.40185236e-24j,
                          -6.20705802e-10 - 4.42467607e-12j],
                         [-6.91213385e-03 + 4.81426465e-21j,
                          -6.91691201e-03 - 1.96203657e-05j]])
    assert np.allclose(data['chiM_qw'], monopole)

    dipole = np.array([[-0.19370323 + 6.04520088e-18j,
                        -0.19385203 - 6.06238802e-04j],
                       [-0.20384696 + 6.32211737e-18j,
                        -0.2040121 - 6.73309535e-04j]])
    assert np.allclose(data['chiD_qw'], dipole)


# test limited features that should work in parallel
@pytest.mark.skipif(size == 1, reason='Features already tested '
                    'in serial in test_basics')
@pytest.mark.skipif(size > 6, reason='Parallelization for '
                    'small test-system broken for many cores')
@pytest.mark.response
def test_bb_parallel(in_tmp_dir, gpw_files):
    df = dielectric(gpw_files['mos2_pw'], 0.1, 0.5)
    bb1 = BuildingBlock('mos2', df)
    bb1.calculate_building_block()
    # Make sure that calculation is finished before loading data file
    world.barrier()
    data = np.load('mos2-chi.npz')
    maxM = np.amax(abs(data['chiM_qw']))
    assert maxM == pytest.approx(0.25076046486693826)
    maxD = np.amax(abs(data['chiD_qw']))
    assert maxD == pytest.approx(0.844873415471949)
