"""This script asserts that the chi's obtained from GS calculations using
symmetries and GS calculations not using symmetries return the same results.
"""

import pytest
import numpy as np

from gpaw.response.chi0 import Chi0
from gpaw.test.gpwfile import response_band_cutoff


@pytest.mark.response
def test_symmetry_si2(gpw_files):
    band_cutoff = response_band_cutoff['fancy_si_pw']
    data_s = []
    for name in ['fancy_si_pw_nosym', 'fancy_si_pw']:
        chi0 = Chi0(gpw_files[name], nbands=band_cutoff)
        data = chi0.calculate([1 / 4, 0, 1 / 4])
        data_s.append(data.chi0_WgG)

        # With a non-Gamma q-point as input, we should therefore
        # not have any data from the optical limit extensions
        assert data.chi0_WxvG is None
        assert data.chi0_Wvv is None

    datadiff_WgG = np.abs(data_s[0] - data_s[1])
    assert datadiff_WgG == pytest.approx(0, abs=1e-6), \
        datadiff_WgG.max()
