import pytest

from gpaw import GPAW
from gpaw.xas import XAS
import gpaw.mpi as mpi


def test_si_paralell(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    if mpi.size == 5:
        rank = mpi.world.rank
        # print(rank)
        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4])
        if rank in [1, 2, 3, 4]:
            calc2 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm2)
            xas2 = XAS(calc2)
            x2, y2 = xas2.get_matrix_element()

            mpi.send((x2, y2), 0, mpi.world)
        elif rank == 0:
            calc1 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm)
            xas1 = XAS(calc1)

            x1, y1 = xas1.get_matrix_element()

            x2, y2 = mpi.receive(1, mpi.world)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)

        # assert 0, 'Ja'

        if rank in [1, 2, 3, 4]:
            calc2 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm2)
            xas2 = XAS(calc2)
            xas2.save_matrix_element('test.npz')
