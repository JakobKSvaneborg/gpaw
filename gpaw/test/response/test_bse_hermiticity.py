"""Test that the BSE Hamiltonian is Hermitian for non-centrosymmetric systems.

MoS2 lacks inversion symmetry, so the screened interaction W_GG'(q) can
have non-zero imaginary parts. When the symmetry operation mapping a BZ
q-vector to the IBZ involves time-reversal (sign == -1), the physical W
at the BZ point is the complex conjugate of W at the IBZ point. Failing
to account for this produces a non-Hermitian BSE matrix.
"""
import numpy as np
import pytest

from gpaw.response.bse import BSE


def _build_bse_matrix(gpwfile, conjugate_W_for_time_reversal):
    """Build the BSE matrix, optionally disabling the W conjugation fix.

    When conjugate_W_for_time_reversal=True, this is the correct behavior.
    When False, it reproduces the old (buggy) behavior where W was never
    conjugated for time-reversed symmetry operations.
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
    return bse_matrix.H_sS


@pytest.mark.response
def test_bse_hermiticity(in_tmp_dir, gpw_files):
    """Check that the TDA BSE matrix H_sS is Hermitian for MoS2."""
    from ase.parallel import world
    if world.size > 1:
        pytest.skip("test_bse_hermiticity can only run in serial mode")
    gpwfile = gpw_files['mos2_5x5_pw']

    # --- 1. With the fix: matrix must be Hermitian ---
    H_fixed = _build_bse_matrix(gpwfile,
                                conjugate_W_for_time_reversal=True)

    deviation_fixed = np.abs(H_fixed - H_fixed.conj().T).max()
    assert deviation_fixed < 1e-12, (
        f'BSE matrix is not Hermitian with fix: '
        f'max|H - H†| = {deviation_fixed:.2e}')

    # --- 2. Without the fix: matrix must NOT be Hermitian ---
    H_broken = _build_bse_matrix(gpwfile,
                                 conjugate_W_for_time_reversal=False)

    deviation_broken = np.abs(H_broken - H_broken.conj().T).max()
    assert deviation_broken > 1e-6, (
        f'Expected non-Hermitian matrix without fix, but '
        f'max|H - H†| = {deviation_broken:.2e} is too small. '
        f'The test may not be exercising time-reversed q-points.')

    # --- 3. The fix and the broken version should differ ---
    diff = np.abs(H_fixed - H_broken).max()
    assert diff > 1e-6, (
        f'Fixed and broken matrices are identical (diff={diff:.2e}). '
        f'The fix may not be active for this system.')
