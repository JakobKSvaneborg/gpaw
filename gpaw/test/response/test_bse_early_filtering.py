"""Test that early transition filtering in BSE produces identical results.

This test verifies that when deps_max is specified:
1. Early filtering (deps_max at BSE construction) produces the same eigenvalues
   as the original approach (deps_max only at diagonalization time).

The early filtering optimization skips computing matrix elements for transitions
that will be filtered out anyway, which can provide significant speedup.
"""
import numpy as np
import pytest
from time import time

from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw.mpi import world
from gpaw.response.bse import BSE


def create_si_gpw(filename):
    """Create a silicon ground state calculation."""
    a = 5.431
    atoms = bulk('Si', 'diamond', a=a)
    calc = GPAW(mode='pw',
                kpts={'size': (2, 2, 2), 'gamma': True},
                occupations=FermiDirac(0.001),
                nbands=12,
                convergence={'bands': -4},
                txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(filename, 'all')
    return filename


def run_bse_with_early_filtering(gpw_file, deps_max, eshift=0.8):
    """Run BSE with early filtering (deps_max at construction time)."""
    bse = BSE(gpw_file,
              ecut=50.,
              valence_bands=range(1, 4),
              conduction_bands=range(4, 8),
              deps_max=deps_max,
              eshift=eshift,
              nbands=8,
              txt=None)
    bse_matrix = bse.get_bse_matrix()
    w_T, v_Rt, exclude_S = bse.diagonalize_bse_matrix(bse_matrix)
    return w_T, exclude_S, bse.nS


def run_bse_without_early_filtering(gpw_file, deps_max, eshift=0.8):
    """Run BSE without early filtering (deps_max only at diagonalization)."""
    bse = BSE(gpw_file,
              ecut=50.,
              valence_bands=range(1, 4),
              conduction_bands=range(4, 8),
              deps_max=None,  # No early filtering
              eshift=eshift,
              nbands=8,
              txt=None)
    bse_matrix = bse.get_bse_matrix()
    # Pass deps_max to diagonalization instead
    w_T, v_Rt, exclude_S = bse_matrix.diagonalize_tammdancoff(
        bse, deps_max=deps_max)
    return w_T, exclude_S, bse.nS


@pytest.mark.response
def test_bse_early_filtering(in_tmp_dir):
    """Test that early filtering produces identical eigenvalues."""
    gpw_file = create_si_gpw('Si.gpw')
    deps_max = 5.0  # eV - should filter out some transitions

    # Run with early filtering
    w_early, exclude_early, nS = run_bse_with_early_filtering(
        gpw_file, deps_max)

    # Run without early filtering
    w_late, exclude_late, _ = run_bse_without_early_filtering(
        gpw_file, deps_max)

    # With reduced basis, exclude_early is empty (filtering already done)
    # With late filtering, exclude_late contains filtered indices
    # The key check is that eigenvalue counts match
    n_kept_early = len(w_early)
    n_kept_late = len(w_late)

    # Verify same number of eigenvalues (same number of kept states)
    assert n_kept_early == n_kept_late, \
        f"Different number of eigenvalues: {n_kept_early} vs {n_kept_late}"

    # Sort eigenvalues for comparison (order may differ slightly)
    w_early_sorted = np.sort(w_early.real)
    w_late_sorted = np.sort(w_late.real)

    np.testing.assert_allclose(
        w_early_sorted, w_late_sorted, rtol=1e-10, atol=1e-12,
        err_msg="Eigenvalues differ between early and late filtering")

    # Print summary
    n_filtered = nS - n_kept_early
    if world.rank == 0:
        print(f"\nBSE Early Filtering Test Results:")
        print(f"  Total pair orbitals: {nS}")
        print(f"  Filtered transitions: {n_filtered}")
        print(f"  Kept for diagonalization: {n_kept_early}")
        print(f"  Lowest eigenvalue: {w_early_sorted[0]:.6f} Ha")
        print(f"  Highest eigenvalue: {w_early_sorted[-1]:.6f} Ha")


@pytest.mark.response
def test_bse_early_filtering_no_exclusion(in_tmp_dir):
    """Test with deps_max large enough to include all transitions."""
    gpw_file = create_si_gpw('Si.gpw')
    deps_max = 100.0  # eV - should include all transitions

    w_early, exclude_early, nS = run_bse_with_early_filtering(
        gpw_file, deps_max)
    w_late, exclude_late, _ = run_bse_without_early_filtering(
        gpw_file, deps_max)

    # With large deps_max, no transitions should be excluded
    assert len(exclude_early) == 0, "Should not exclude any transitions"
    assert len(exclude_late) == 0, "Should not exclude any transitions"

    # Eigenvalues should still match
    np.testing.assert_allclose(
        np.sort(w_early.real), np.sort(w_late.real),
        rtol=1e-10, atol=1e-12)


def main():
    """Run tests with timing information when executed as main."""
    import tempfile
    import os

    print("=" * 60)
    print("BSE Early Filtering Validation Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)

        # Create ground state
        print("\n1. Creating Si ground state calculation...")
        t0 = time()
        gpw_file = create_si_gpw('Si.gpw')
        t_gs = time() - t0
        print(f"   Ground state time: {t_gs:.2f} s")

        deps_max = 5.0  # eV

        # Run with early filtering
        print(f"\n2. Running BSE with early filtering (deps_max={deps_max} eV)...")
        t0 = time()
        w_early, exclude_early, nS = run_bse_with_early_filtering(
            gpw_file, deps_max)
        t_early = time() - t0
        print(f"   Early filtering time: {t_early:.2f} s")

        # Run without early filtering
        print(f"\n3. Running BSE without early filtering...")
        t0 = time()
        w_late, exclude_late, _ = run_bse_without_early_filtering(
            gpw_file, deps_max)
        t_late = time() - t0
        print(f"   Late filtering time: {t_late:.2f} s")

        # Verify results match
        print("\n4. Comparing results...")
        w_early_sorted = np.sort(w_early.real)
        w_late_sorted = np.sort(w_late.real)

        # Check eigenvalue counts match first
        if len(w_early) != len(w_late):
            print(f"   ERROR: Different eigenvalue counts: {len(w_early)} vs {len(w_late)}")
            return 1

        max_diff = np.max(np.abs(w_early_sorted - w_late_sorted))
        match = np.allclose(w_early_sorted, w_late_sorted, rtol=1e-10, atol=1e-12)

        n_filtered = nS - len(w_early)
        print(f"   Total pair orbitals: {nS}")
        print(f"   Filtered transitions: {n_filtered}")
        print(f"   Eigenvalues computed: {len(w_early)}")
        print(f"   Maximum eigenvalue difference: {max_diff:.2e}")
        print(f"   Results match: {match}")

        # Timing summary
        print("\n" + "=" * 60)
        print("Timing Summary")
        print("=" * 60)
        print(f"  Ground state:        {t_gs:8.2f} s")
        print(f"  Early filtering BSE: {t_early:8.2f} s")
        print(f"  Late filtering BSE:  {t_late:8.2f} s")
        if t_late > 0:
            speedup = t_late / t_early
            print(f"  Speedup factor:      {speedup:8.2f}x")
        print("=" * 60)

        if not match:
            print("\nERROR: Results do not match!")
            return 1

        print("\nSUCCESS: Early filtering produces identical results.")
        return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
