import pytest
import numpy as np
from gpaw.nlopt.linear import get_chi_tensor
from gpaw.nlopt.matrixel import make_nlodata
from gpaw.mpi import world


@pytest.mark.skipif(world.size > 4, reason='System too small')
def test_shift(in_tmp_dir, gpw_files):
    # Get the mml
    nlodata = make_nlodata(gpw_files['h_shift'])

    # Do a linear response caclulation
    freqs = np.linspace(0, 5, 101)
    get_chi_tensor(nlodata, freqs=freqs, out_name='linear.npy')

    # Check it
    if world.rank == 0:
        chi = np.load('linear.npy')

        # Check for nan's
        assert not np.isnan(chi).any()
