import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.new.rttddft import RTTDDFT
from gpaw.new.rttddft.writers import DipoleMomentWriter
from gpaw.tddft.spectrum import read_dipole_moment_file
from gpaw.tddft.units import asetime_to_autime

from gpaw.test import only_on_master


@pytest.fixture(scope='module')
def nacl_nospin(gpw_files):
    return gpw_files['nacl_nospin']


@pytest.mark.skipif(world.size > 1,
                    reason='Only serial execution supported in new RTDDFT')
@pytest.mark.ci
def test_dipolemoment(nacl_nospin, in_tmp_dir):
    td_calc = RTTDDFT.from_file(nacl_nospin, td_algorithm='ecn')
    dt = 1e-3

    with DipoleMomentWriter('dm.dat') as dmwriter:
        # Write the start comment
        dmwriter.write_comment(f'Start at {td_calc.time:.8f}')

        # Kick, write a comment about the kick, and write the dipole moment
        td_calc.absorption_kick([0.0, 0.0, 1e-5])
        dmwriter.write_kick(td_calc.history.most_recent_kick)
        dmwriter.write_dm(td_calc.time, td_calc.state, td_calc.pot_calc)

        # Propagate
        for _ in td_calc.ipropagate(dt, 3):
            dmwriter.write_dm(td_calc.time, td_calc.state, td_calc.pot_calc)

    # Read dipole moment file
    world.barrier()
    kick_i, time1_t, _, dm1_tv = read_dipole_moment_file('dm.dat')

    assert len(kick_i) == 1
    np.testing.assert_almost_equal(kick_i[0]['time'], 0)
    np.testing.assert_allclose(kick_i[0]['strength_v'], [0, 0, 1e-5])

    # Restart in append mode
    with DipoleMomentWriter('dm.dat', append=True) as dmwriter:
        # Write the start comment
        dmwriter.write_comment(f'Start at {td_calc.time:.8f}')

        # Kick twice and write the dipole moment
        td_calc.absorption_kick([0.0, 0.0, 2e-5])
        dmwriter.write_kick(td_calc.history.most_recent_kick)
        td_calc.absorption_kick([0.0, 0.0, 3e-5])
        dmwriter.write_kick(td_calc.history.most_recent_kick)
        dmwriter.write_dm(td_calc.time, td_calc.state, td_calc.pot_calc)

    # Read dipole moment file
    world.barrier()
    kick_i, time2_t, _, dm2_tv = read_dipole_moment_file('dm.dat')

    # Make sure that the first data are the same
    np.testing.assert_allclose(time2_t[:4], time1_t)
    np.testing.assert_allclose(dm2_tv[:4], dm1_tv)

    # Make sure that all three kicks are there
    assert len(kick_i) == 3
    kick_time = 3 * dt * asetime_to_autime
    np.testing.assert_almost_equal(kick_i[0]['time'], 0)
    np.testing.assert_almost_equal(kick_i[1]['time'], kick_time)
    np.testing.assert_almost_equal(kick_i[2]['time'], kick_time)
    np.testing.assert_allclose(kick_i[0]['strength_v'], [0, 0, 1e-5])
    np.testing.assert_allclose(kick_i[1]['strength_v'], [0, 0, 2e-5])
    np.testing.assert_allclose(kick_i[2]['strength_v'], [0, 0, 3e-5])

    # Start over
    with DipoleMomentWriter('dm.dat') as dmwriter:
        # We need to write at least two entries, the helper function cannot
        # read the file otherwise
        dmwriter.write_dm(td_calc.time, td_calc.state, td_calc.pot_calc)
        # Propagate once
        for _ in td_calc.ipropagate(dt, 1):
            pass
        dmwriter.write_dm(td_calc.time, td_calc.state, td_calc.pot_calc)

    # Read dipole moment file
    world.barrier()
    kick_i, time3_t, _, dm3_tv = read_dipole_moment_file('dm.dat')

    assert len(kick_i) == 0  # There should be no kicks

    # The first data should be identical to the last data from the
    # previous file since we did not propagate inbetween
    np.testing.assert_allclose(time2_t[-1], time3_t[0])
    np.testing.assert_allclose(dm2_tv[-1], dm3_tv[0])


@only_on_master(world)
@pytest.mark.ci
def test_parse_version_1(in_tmp_dir):
    # Write old-style dipole moment file, filled with random data
    kick = (1e-5, 3.333e-3, 4.321e-1)
    data = np.random.rand(3, 5)
    data[:, 0] = np.linspace(0, 1, len(data))  # make sure increasing
    with open('dm.dat', 'w') as fp:
        fp.write('''
# DipoleMomentWriter[version=1](center=False, density='comp')''')
        fp.write('''
# Kick = [%22.12le, %22.12le, %22.12le]; ''' % tuple(kick))
        fp.write('Gauge = length; Time = 0.00000000')
        for row in data:
            fp.write('''
%20.8lf %22.12le %22.12le %22.12le %22.12le''' % tuple(row))

    # Read the file
    kick_i, time_t, norm_t, dm_tv = read_dipole_moment_file('dm.dat')

    # Check that the kick is correct
    assert len(kick_i) == 1
    np.testing.assert_almost_equal(kick_i[0]['time'], 0)
    np.testing.assert_allclose(kick_i[0]['strength_v'], kick)

    # Verify the data
    np.testing.assert_allclose(time_t, data[:, 0])
    np.testing.assert_allclose(norm_t, data[:, 1])
    np.testing.assert_allclose(dm_tv, data[:, 2:])
