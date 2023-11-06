import numpy as np
import pytest

import gpaw.mpi as mpi
from gpaw import GPAW
from gpaw.test import equal
from gpaw.xas import XAS

@pytest.mark.later
def test_Mg_2s_xas(in_tmp_dir, gpw_files):
    calc = GPAW(gpw_files['Mg_2p_xas'])
    if mpi.size == 1:
        xas = XAS(calc)
        x_s, y_s = xas.get_spectra(stick=True)
        e1_n = xas.eps_n
        dks = 94.17740184323611
        shift = dks - x_s[0]
        y_tot_s = np.zeros(y_s.shape[-1])
        for m in range(1):
            y_tot_s += (y_s[0,m] + y_s[1,m] + y_s[2,m])
        
        for ie, e in enumerate(x_s):
            if y_tot_s[ie] > 10**(-5):
                break
        print(y_tot_s)
        assert e + shift == dks
        
