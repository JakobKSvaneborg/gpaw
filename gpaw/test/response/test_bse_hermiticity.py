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


def _build_bse_matrix(gpwfile, conjugate_W_for_time_reversal, comm):
    """Build the BSE matrix, optionally disabling the W conjugation fix.

    When conjugate_W_for_time_reversal=True, this is the correct behavior.
    When False, it reproduces the old (buggy) behavior where W was never
    conjugated for time-reversed symmetry operations.
    """
    bse = BSE(
        gpwfile,
        ecut=50,
        valence_bands=2,
        conduction_bands=2,
        nbands=16,
        mode="BSE",
        truncation="2D",
        comm=comm,
    )

    if not conjugate_W_for_time_reversal:
        # Monkeypatch get_density_matrix to always report sign=+1,
        # so add_direct_kernel never conjugates W.  This reproduces
        # the old behavior before the fix.
        _orig = bse.get_density_matrix

        def _patched(*args, **kwargs):
            rho, iq, _sign = _orig(*args, **kwargs)
            return rho, iq, 1  # force sign=+1

        bse.get_density_matrix = _patched

    bse_matrix = bse.get_bse_matrix(optical=True)
    return bse, bse_matrix.H_sS


@pytest.mark.response
def test_bse_hermiticity(in_tmp_dir, gpw_files, mpi):
    """Check that the TDA BSE matrix H_sS is Hermitian for MoS2."""

    gpwfile = gpw_files["mos2_5x5_pw"]
    comm = mpi.comm

    bse_h, H_hermitian_sS = _build_bse_matrix(
        gpwfile, conjugate_W_for_time_reversal=True, comm=comm,
    )

    # Original, bugged BSE matrix
    bse_o, H_original_sS = _build_bse_matrix(
        gpwfile, conjugate_W_for_time_reversal=False, comm=comm,
    )

    H_hermitian_SS = bse_h.collect_A_SS(H_hermitian_sS)
    H_original_SS = bse_o.collect_A_SS(H_original_sS)

    if comm.rank == 0:
        deviation_h = np.abs(H_hermitian_SS - H_hermitian_SS.conj().T).max()
        deviation_o = np.abs(H_original_SS - H_original_SS.conj().T).max()
    else:
        deviation_h = 0.0
        deviation_o = 0.0
    deviation_o = broadcast_float(deviation_o, comm)
    deviation_h = broadcast_float(deviation_h, comm)

    assert deviation_h < 1e-12, (
        "BSE matrix is not Hermitian with fix: "
        f"max|H - H†| = {deviation_h:.2e}"
    )

    assert deviation_o > 1e-6, (
        f"Expected non-Hermitian matrix without fix, but "
        f"max|H - H†| = {deviation_o:.2e} is too small. "
        f"The test may not be exercising time-reversed q-points."
    )
