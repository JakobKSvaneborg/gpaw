"""Test the site Zeeman energy sum rules."""

import numpy as np
import pytest


def test():
    # Load data
    rc_r = np.load('rc_r.npy')
    EZ_r = np.load('EZ_r.npy')
    sp_EZ_r = np.load('sp_EZ_r.npy')
    EZ_nr = np.load('EZ_nr.npy')

    # Test the single-particle sum rule
    assert sp_EZ_r == pytest.approx(EZ_r, abs=0.01)

    # Test the two-particle sum rule
    # First, check that all values are positive and real
    assert np.all(EZ_nr.real > 0.)
    assert EZ_nr.imag == pytest.approx(0.)
    tp_EZ_nr = EZ_nr.real
    # Secondly, test accuracy for r > 1.1 Å, excluding unocc∈{0, 4, 8}
    rmask = rc_r > 1.1
    for tp_EZ_r in tp_EZ_nr[3:]:
        assert tp_EZ_r[rmask] == pytest.approx(EZ_r[rmask], abs=0.025)
    # Finally, test that there exist a hierachy of site Zeeman energy values,
    # when increasing the number of bands for 0.2 Å < r < 0.8 Å
    rmask = np.logical_and(0.2 < rc_r, rc_r < 0.8)
    for n in range(1, tp_EZ_nr.shape[0]):
        assert np.all(tp_EZ_nr[n - 1, rmask] < tp_EZ_nr[n, rmask])
    assert np.all(tp_EZ_nr[-1, rmask] < EZ_r[rmask])


if __name__ == '__main__':
    test()
