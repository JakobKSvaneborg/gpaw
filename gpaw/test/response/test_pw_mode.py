"""Test that pw_mode (direct reciprocal-space pair density) gives the same
chi0 as the standard FFT-based approach."""
import pytest
import numpy as np
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import serial_comm
from gpaw.response.chi0 import Chi0Calculator
from gpaw.response.context import ResponseContext
from gpaw.response.frequencies import FrequencyDescriptor


@pytest.mark.response
def test_pw_mode_vs_fft(in_tmp_dir, mpi):
    """Verify that pw_mode produces the same chi0 as the FFT approach."""
    a = bulk('Si', 'diamond')

    calc = a.calc = mpi.GPAW(
        kpts={'size': (2, 2, 2), 'gamma': True},
        symmetry={'point_group': True},
        mode=PW(150),
        occupations=FermiDirac(width=0.001),
        convergence={'bands': 8},
        txt='si_gs.txt')
    a.get_potential_energy()
    calc.write('si.gpw', 'all')

    calc = GPAW('si.gpw', txt=None, communicator=serial_comm)

    q_c = [0, 0, 0.5]
    wd = FrequencyDescriptor.from_array_or_dict([0, 1.0, 2.0])

    # Standard FFT-based chi0
    context_fft = ResponseContext(txt='chi0_fft.log', comm=mpi.comm)
    chi0_calc_fft = Chi0Calculator(
        gs=calc, context=context_fft,
        wd=wd, hilbert=False, ecut=50, pw_mode=False)
    chi0_fft = chi0_calc_fft.calculate(q_c)
    chi0_fft_wGG = chi0_fft.chi0_WgG.copy()

    # Direct reciprocal-space chi0
    context_pw = ResponseContext(txt='chi0_pw.log', comm=mpi.comm)
    chi0_calc_pw = Chi0Calculator(
        gs=calc, context=context_pw,
        wd=wd, hilbert=False, ecut=50, pw_mode=True)
    chi0_pw = chi0_calc_pw.calculate(q_c)
    chi0_pw_wGG = chi0_pw.chi0_WgG.copy()

    # They should agree to high precision
    err = np.abs(chi0_fft_wGG - chi0_pw_wGG).max()
    assert err < 1e-10, \
        f'pw_mode chi0 differs from FFT chi0 by {err}'
