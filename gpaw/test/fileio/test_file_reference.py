"""Test the reading of wave functions as file references."""
import numpy as np

from gpaw import GPAW


def test_fileio_file_reference(in_tmp_dir, gpw_files, mpi):
    # load restart from gpw
    calc = mpi.GPAW(gpw_files['na3_fd_kp_restart'])
    wf0 = calc.get_pseudo_wave_function(2, 1, 1)

    # Now read with a single process
    comm = mpi.comm.new_communicator(np.array((mpi.comm.rank,)))
    calc = GPAW(gpw_files['na3_fd_kp_restart'], communicator=comm)
    wf1 = calc.get_pseudo_wave_function(2, 1, 1)

    # compare wf restarts match
    diff = np.abs(wf0 - wf1)
    assert np.all(diff < 1e-12)
