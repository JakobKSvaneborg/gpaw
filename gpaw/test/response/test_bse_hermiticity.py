"""Test that the BSE Hamiltonian is Hermitian for non-centrosymmetric systems.

MoS2 lacks inversion symmetry, so the screened interaction W_GG'(q) can
have non-zero imaginary parts. When the symmetry operation mapping a BZ
q-vector to the IBZ involves time-reversal (sign == -1), the physical W
at the BZ point is the complex conjugate of W at the IBZ point. Failing
to account for this produces a non-Hermitian BSE matrix.
"""
import numpy as np
import pytest

from gpaw.mpi import broadcast_float
from gpaw.response.bse import BSE


def _build_bse_matrix(gpwfile, conjugate_W_for_time_reversal):
    """Build the BSE matrix, optionally disabling the W conjugation fix.

    When conjugate_W_for_time_reversal=True, this is the correct behavior.
    When False, it reproduces the old (buggy) behavior where W was never
    conjugated for time-reversed symmetry operations.

    Returns the BSE object and the (distributed) H_sS matrix.
    """
    bse = BSE(gpwfile,
              ecut=10,
              valence_bands=2,
              conduction_bands=2,
              nbands=15,
              mode='BSE',
              truncation='2D')

    if not conjugate_W_for_time_reversal:
        # Monkeypatch get_density_matrix to always report sign=+1,
        # so add_direct_kernel never conjugates W.  This reproduces
        # the old behavior before the fix.
        _orig = bse.get_density_matrix

        def _patched(*args, **kwargs):
            rho, iq, _sign = _orig(*args, **kwargs)
            return rho, iq, 1          # force sign=+1

        bse.get_density_matrix = _patched

    bse_matrix = bse.get_bse_matrix(optical=True)
    return bse, bse_matrix.H_sS


@pytest.mark.response
def test_bse_hermiticity(in_tmp_dir, gpw_files):
    """Check that the TDA BSE matrix H_sS is Hermitian for MoS2."""
    gpwfile = gpw_files['mos2_5x5_pw']

    # Build both matrices and gather from distributed to full on rank 0
    bse_fixed, H_fixed_sS = _build_bse_matrix(
        gpwfile, conjugate_W_for_time_reversal=True)
    bse_broken, H_broken_sS = _build_bse_matrix(
        gpwfile, conjugate_W_for_time_reversal=False)

    # Gather distributed matrices to rank 0 and compute deviations
    comm = bse_fixed.context.comm
    H_fixed = bse_fixed.collect_A_SS(H_fixed_sS)
    H_broken = bse_broken.collect_A_SS(H_broken_sS)

    if comm.rank == 0:
        deviation_fixed = np.abs(H_fixed - H_fixed.conj().T).max()
        deviation_broken = np.abs(H_broken - H_broken.conj().T).max()
        diff = np.abs(H_fixed - H_broken).max()
    else:
        deviation_fixed = deviation_broken = diff = 0.0

    # Broadcast so all ranks assert together (avoids MPI deadlock on failure)
    deviation_fixed = broadcast_float(deviation_fixed, comm)
    deviation_broken = broadcast_float(deviation_broken, comm)
    diff = broadcast_float(diff, comm)

    # --- 1. With the fix: matrix must be Hermitian ---
    assert deviation_fixed < 1e-12, (
        f'BSE matrix is not Hermitian with fix: '
        f'max|H - H†| = {deviation_fixed:.2e}')

    # --- 2. Without the fix: matrix must NOT be Hermitian ---
    assert deviation_broken > 1e-6, (
        f'Expected non-Hermitian matrix without fix, but '
        f'max|H - H†| = {deviation_broken:.2e} is too small. '
        f'The test may not be exercising time-reversed q-points.')

    # --- 3. The fix and the broken version should differ ---
    assert diff > 1e-6, (
        f'Fixed and broken matrices are identical (diff={diff:.2e}). '
        f'The fix may not be active for this system.')
