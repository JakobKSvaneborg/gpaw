import numpy as np
import pytest
from ase.dft.kpoints import monkhorst_pack
from gpaw.response.kpoints import KPointFinder

def test_kpoint_finder_rounding_stability():
    """
    Regression test for issue where Nk=128 caused KPointFinder failures
    due to rounding instability at 6 decimal places.
    1/128 = 0.0078125, which is exactly at the tie-breaking point for
    rounding to 6 decimals. Small noise could cause it to round differently.
    """
    Nk = 128

    # Generate Gamma-centered Monkhorst-Pack grid
    kpts = monkhorst_pack((Nk, Nk, 1))
    # Shift to center at Gamma if Nk is even (monkhorst_pack is shifted by 0.5/N for even N)
    shift = np.array([0.5/Nk, 0.5/Nk, 0]) if Nk % 2 == 0 else np.zeros(3)
    kpts += shift

    # Introduce small noise to kpts to simulate numerical inaccuracies
    # (e.g. from symmetry operations or file IO)
    # The noise should be small (e.g. 1e-12), but enough to trigger rounding flip
    # if the precision is insufficient.
    rng = np.random.RandomState(42)
    noise = (rng.rand(*kpts.shape) - 0.5) * 1e-12
    kpts_noisy = kpts + noise

    # Construct KPointFinder with noisy kpoints
    finder = KPointFinder(kpts_noisy)

    # Generate q-points that are exactly on grid (clean)
    q_qc = np.zeros((Nk, 3))
    q_qc[:, 0] = np.linspace(0, 1, Nk, endpoint=False)

    # Check compatibility
    # We check if (k_noisy[0] + q_clean) is found in the grid.
    # KPointFinder.find uses the internal rounding logic.

    k_source = kpts_noisy[0]

    failures = 0
    max_dist = 0.0

    for i in range(len(q_qc)):
        q = q_qc[i]
        target = k_source + q

        # We expect this to succeed because the grid is uniform.
        # The finder should map 'target' to some point in 'kpts_noisy'
        # despite the noise and the rounding boundary.

        # KPointFinder.find raises ValueError if not found
        try:
            k_index = finder.find(target)
        except ValueError:
            failures += 1
            # Calculate distance manually to debug/assert
            # Replicate finding logic partially to see distance
            target_rounded = finder._round(target)
            dist, _ = finder.kdtree.query(target_rounded)
            if dist > max_dist:
                max_dist = dist

    assert failures == 0, (f"KPointFinder failed for {failures}/{len(q_qc)} points with Nk={Nk}. "
                           f"Max distance found: {max_dist}")
