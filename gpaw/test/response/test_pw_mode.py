"""Test that the coarse-grid pair density (ecut_pair) gives results
converging toward the standard FFT-based approach."""
import pytest
import numpy as np
from ase.build import bulk

from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import serial_comm
from gpaw.response.chi0 import Chi0Calculator
from gpaw.response.context import ResponseContext
from gpaw.response.frequencies import FrequencyDescriptor


@pytest.mark.response
def test_ecut_pair_convergence(in_tmp_dir, mpi):
    """Verify that ecut_pair results converge to the exact (full-grid) result.

    With ecut_pair = ecut_gs, the truncation includes all plane waves and
    the result should match the standard calculation to high precision
    (only aliasing differences remain, which are identical in character).
    """
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
    context_ref = ResponseContext(txt='chi0_ref.log', comm=mpi.comm)
    chi0_ref = Chi0Calculator(
        gs=calc, context=context_ref,
        wd=wd, hilbert=False, ecut=50).calculate(q_c)
    ref_wGG = chi0_ref.chi0_WgG.copy()

    # Coarse-grid chi0 with ecut_pair = ecut_gs (should be near-exact)
    context_full = ResponseContext(txt='chi0_full.log', comm=mpi.comm)
    chi0_full = Chi0Calculator(
        gs=calc, context=context_full,
        wd=wd, hilbert=False, ecut=50, ecut_pair=150).calculate(q_c)
    full_wGG = chi0_full.chi0_WgG.copy()

    err_full = np.abs(ref_wGG - full_wGG).max()
    scale = np.abs(ref_wGG).max()
    assert err_full / scale < 1e-4, \
        f'ecut_pair=ecut_gs differs from reference by {err_full/scale:.2e}'

    # Coarse-grid chi0 with smaller ecut_pair (approximate, but converging)
    context_approx = ResponseContext(txt='chi0_approx.log', comm=mpi.comm)
    chi0_approx = Chi0Calculator(
        gs=calc, context=context_approx,
        wd=wd, hilbert=False, ecut=50, ecut_pair=100).calculate(q_c)
    approx_wGG = chi0_approx.chi0_WgG.copy()

    err_approx = np.abs(ref_wGG - approx_wGG).max()
    # Should be reasonably close (truncation error is small for Si)
    assert err_approx / scale < 0.05, \
        f'ecut_pair=100 differs from reference by {err_approx/scale:.2e}'

    # The full-cutoff result should be closer than the truncated one
    assert err_full < err_approx, \
        'Larger ecut_pair should give better accuracy'
