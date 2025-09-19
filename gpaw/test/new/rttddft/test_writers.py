import numpy as np
import pytest

from gpaw.mpi import world
from gpaw.new.rttddft import RTTDDFT
from gpaw.new.rttddft.writers import DipoleMomentWriter
from gpaw.tddft.spectrum import read_dipole_moment_file
from gpaw.tddft.units import asetime_to_autime


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
        dmwriter.write_start(td_calc.history)

        # Kick and write the dipole moment
        td_calc.absorption_kick([0.0, 0.0, 1e-5])
        dmwriter.write_dm(td_calc.history, td_calc.state, td_calc.pot_calc)

        # Propagate
        for _ in td_calc.ipropagate(dt, 3):
            dmwriter.write_dm(td_calc.history, td_calc.state, td_calc.pot_calc)

    # Read dipole moment file
    world.barrier()
    kick_i, time1_t, _, dm1_tv = read_dipole_moment_file('dm.dat')

    assert len(kick_i) == 1
    np.testing.assert_almost_equal(kick_i[0]['time'], 0)
    np.testing.assert_allclose(kick_i[0]['strength_v'], [0, 0, 1e-5])

    # Restart in append mode
    with DipoleMomentWriter('dm.dat', append=True) as dmwriter:
        # Write the start comment
        dmwriter.write_start(td_calc.history)

        # Kick twice and write the dipole moment
        td_calc.absorption_kick([0.0, 0.0, 2e-5])
        td_calc.absorption_kick([0.0, 0.0, 3e-5])
        dmwriter.write_dm(td_calc.history, td_calc.state, td_calc.pot_calc)

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
        dmwriter.write_dm(td_calc.history, td_calc.state, td_calc.pot_calc)
        # Propagate once
        for _ in td_calc.ipropagate(dt, 1):
            pass
        dmwriter.write_dm(td_calc.history, td_calc.state, td_calc.pot_calc)

    # Read dipole moment file
    world.barrier()
    kick_i, time3_t, _, dm3_tv = read_dipole_moment_file('dm.dat')

    # The last two kicks should have been written, because we have not
    # propaged after them. This behavior is unintuitive perhaps, but
    # stopping the time propagation right after a kick is also a bit weird.
    assert len(kick_i) == 2
    np.testing.assert_almost_equal(kick_i[0]['time'], kick_time)
    np.testing.assert_almost_equal(kick_i[1]['time'], kick_time)
    np.testing.assert_allclose(kick_i[0]['strength_v'], [0, 0, 2e-5])
    np.testing.assert_allclose(kick_i[1]['strength_v'], [0, 0, 3e-5])

    # The first data should be identical to the last data from the
    # previous file since we did not propagate inbetween
    np.testing.assert_allclose(time2_t[-1], time3_t[0])
    np.testing.assert_allclose(dm2_tv[-1], dm3_tv[0])
