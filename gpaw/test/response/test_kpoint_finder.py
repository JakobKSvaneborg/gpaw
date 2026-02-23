import numpy as np
import pytest
from ase.dft.kpoints import monkhorst_pack
from gpaw.response.kpoints import KPointFinder


@pytest.mark.parametrize('Nk', [10, 101, 128])
def test_kpoint_finder_rounding_stability(Nk):
    """
    Test that KpointFinder is robust against rounding errors.
    Previous version would fail with e.g. Nk = 128
    1/128 = 0.0078125, which means that noise
    could cause it to round differently.
    """

    # generate Gamma-centered grid
    kpt_kc = monkhorst_pack((Nk, Nk, 1))
    if not (Nk % 2):
        shift = np.array([0.5 / Nk, 0.5 / Nk, 0])
        kpt_kc += shift
    knorm_k = np.linalg.norm(kpt_kc, axis=1)
    assert np.any(knorm_k < 1e-8)  # confirm that Gamma is in our grid

    finder = KPointFinder(kpt_kc)

    # generate q-points on grid, but with small noise
    # could happen in a real calculation e.g. from symmetry operations
    q_qc = np.zeros((Nk, 3))
    q_qc[:, 0] = np.linspace(0, 1, Nk, endpoint=False)
    rng = np.random.RandomState(42)
    noise_qc = (rng.rand(*q_qc.shape) - 0.5) * 1e-10
    q_qc = q_qc + noise_qc

    # check if q-point can be found on grid
    k0_c = kpt_kc[0]

    failures = 0
    max_dist = 0.0

    for i, q_c in enumerate(q_qc):
        kq_c = k0_c + q_c
        try:
            finder.find(kq_c)
        except ValueError:
            failures += 1
            # calculate distance manually to debug/assert
            dist, _ = finder.kdtree.query(kq_c)
            if dist > max_dist:
                max_dist = dist

    assert failures == 0, (
        f'KPointFinder failed for {failures}/{len(q_qc)} points with Nk={Nk}. '
        f'Max distance found: {max_dist}')

    # test that k-point is correctly wrapped inside BZ
    k = finder.find(np.array([[0.99999999999999, 2, 3]]))
    assert np.allclose(kpt_kc[k], np.zeros(3))
