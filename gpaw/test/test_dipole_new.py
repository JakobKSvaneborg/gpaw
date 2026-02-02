import numpy as np
import pytest

from gpaw.core import UGDesc


def test_ug_moment(mpi):
    ug = UGDesc.from_cell_and_grid_spacing(
        [1.0, 2.0, 3.0, 80, 90, 70],
        0.1,
        comm=mpi.comm,
        pbc=False)
    a = ug.zeros()
    if mpi.comm.rank == 0:
        a.data[3, 4, 5] = 1.0
        mom1_v = ug.xyz()[3, 4, 5] * ug.dv
    else:
        mom1_v = np.empty(3)
    mpi.comm.broadcast(mom1_v, 0)
    mom2_v = a.moment()
    assert mom1_v == pytest.approx(mom2_v)


def test_dipole(gpw_files, mpi):
    calc = mpi.GPAW(gpw_files['co_lcao'])
    d_v = calc.get_atoms().get_dipole_moment()
    print(d_v)
