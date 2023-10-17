import pytest
from ase.units import Bohr

from gpaw.mpi import serial_comm
from gpaw.new.ase_interface import GPAW
from gpaw.nlopt.matrixel import WaveFunctionAdapter


def test_adapter_pseudo_wfs(gpw_files):

    calc = GPAW(gpw_files['sic_pw'], communicator=serial_comm)

    u_R_fromcalc = calc.get_pseudo_wave_function(3, 2, 0, periodic=True)
    u_R_fromcalc *= Bohr**1.5

    gs = WaveFunctionAdapter(calc)
    u_R = gs.get_pseudo_wave_function(ni=3, nf=4, k_ind=2, spin=0)[0]

    assert u_R == pytest.approx(u_R_fromcalc, 1e-10)
