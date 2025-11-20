import numpy as np
import pytest

from gpaw.core import PWDesc
from gpaw.core.atom_arrays import AtomArraysLayout
from gpaw.gpu import cupy as cp
from gpaw.gpu.mpi import CuPyMPI
from gpaw.mpi import world
from gpaw.new.c import GPU_AWARE_MPI
from gpaw.new.pwfd.move_wfs import move_wave_functions
from gpaw.setup import create_setup


@pytest.mark.parametrize('xp',
                         [np,
                          pytest.param(cp, marks=pytest.mark.gpu)])
def test_move(xp):
    comm = world if GPU_AWARE_MPI else CuPyMPI(world)
    pw = PWDesc(ecut=25, cell=[2, 2, 2], kpt=[0.25, 0.25, 0.0], comm=comm)
    psit_nG = pw.zeros(2, xp=xp)
    print(psit_nG.data.shape)
    pos1 = np.zeros((1, 3))
    pos2 = pos1 + 0.1
    setup = create_setup('H')
    P_ani = AtomArraysLayout([(5,)], comm, complex, xp).empty(2)
    if comm.rank == 0:
        P_ani[0][:] = 1 + 2j
    move_wave_functions(pos1, pos2, P_ani, psit_nG, [setup])
    move_wave_functions(pos2, pos1, P_ani, psit_nG, [setup])
    assert abs(psit_nG.to_xp(np).data).max() < 1e-12


test_move(np)
