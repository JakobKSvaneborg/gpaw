import pytest

import numpy as np

from gpaw.nlopt.matrixel import make_nlodata
from gpaw.mpi import world


@pytest.mark.skipif(world.size > 1, reason='Only serial')
def test_mme_Ni(gpw_files):

    # Collinear calculation
    nlodata = make_nlodata(gpw_files['fcc_Ni_col'], ni=0, nf=3).distribute()

    data1 = nlodata[34]  # k = (0.5, 0.5, 0.25), s = 0
    E1_col_n = data1[2]
    p1_col_vnn = np.abs(data1[3])

    data2 = nlodata[70]  # k = (0.5, 0.5, 0.25), s = 1
    E2_col_n = data2[2]
    p2_col_vnn = np.abs(data2[3])

    # Noncollinear calculation
    nlodata = make_nlodata(gpw_files['fcc_Ni_ncol'], ni=0, nf=6).distribute()

    data = nlodata[62]  # k = (0.5, 0.5, 0.25), s = 0
    E_ncol_n = data[2]
    p_ncol_vnn = np.abs(data[3])

    # print(np.abs(p1_col_vnn[0]))
    # print(' ')
    # print(np.abs(p2_col_vnn[0]))
    # print(' ')
    # print(np.abs(p_ncol_vnn[0].round(10)))

    assert E_ncol_n[0:3] == pytest.approx(E1_col_n, abs=1.e-9)
    assert E_ncol_n[3:6] == pytest.approx(E2_col_n, abs=1.e-9)
    assert p_ncol_vnn[:, 0:3, 0:3] == pytest.approx(p1_col_vnn, abs=1.e-9)
    assert p_ncol_vnn[:, 3:6, 3:6] == pytest.approx(p2_col_vnn, abs=1.e-9)
