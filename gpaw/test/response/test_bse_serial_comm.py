import numpy as np
import pytest
from unittest.mock import patch

from gpaw.mpi import serial_comm
from gpaw.response.bse import BSE


@pytest.mark.response
def test_bse_diagonalize_with_serial_comm(in_tmp_dir, gpw_files):
    """Test that BSE diagonalization works with serial_comm (no BLACS)."""
    bse = BSE(gpw_files['si_gw_a0_all'],
              ecut=50.,
              valence_bands=3,
              conduction_bands=3,
              deps_max=6,
              eshift=0.8,
              nbands=8,
              comm=serial_comm)
    bse_matrix = bse.get_bse_matrix()

    with patch('gpaw.response.bse.BlacsGrid') as mock_blacs:
        w_T, v_Rt, exclude_S = bse.diagonalize_bse_matrix(bse_matrix)
        # BlacsGrid should never be called with serial comm
        mock_blacs.assert_not_called()

    assert len(exclude_S) == 27
    assert len(w_T) > 0


@pytest.mark.response
def test_bse_get_dielectric_function_serial_comm(in_tmp_dir, gpw_files):
    """Test that BSE.get_dielectric_function works with serial_comm."""
    bse = BSE(gpw_files['si_gw_a0_all'],
              ecut=50.,
              valence_bands=3,
              conduction_bands=3,
              eshift=0.8,
              nbands=8,
              comm=serial_comm)

    # This should complete without errors about BLACS/serial comm
    w_w, eps_w = bse.get_dielectric_function(
        filename=None,
        eta=0.2,
        w_w=np.linspace(0, 10, 101))
    assert not np.all(eps_w.imag == 0)
