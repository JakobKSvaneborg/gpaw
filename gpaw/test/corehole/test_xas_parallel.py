import pytest
from ase.build import bulk

from gpaw import GPAW, FermiDirac
from gpaw.xas import XAS
import gpaw.mpi as mpi
from gpaw.test import gen


def test_si_xas_paralell(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    if mpi.size == 5:
        rank = mpi.world.rank

        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4])
        if rank in [1, 2, 3, 4]:
            calc2 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm2)
            xas2 = XAS(calc2)
            x2, y2 = xas2.get_matrix_element()

            mpi.send((x2, y2), 0, mpi.world)
            xas2.write('test.npz')
        elif rank == 0:
            calc1 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm)
            xas1 = XAS(calc1)

            x1, y1 = xas1.get_matrix_element()

            x2, y2 = mpi.receive(1, mpi.world)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)

def test_si_xas_paralell_nosym(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    if mpi.size == 5:
        rank = mpi.world.rank
        print(rank)
        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4])
        if rank in [1, 2, 3, 4]:
            calc2 = GPAW(gpw_files['si_corehole_nosym_pw'], communicator=comm2)
            xas2 = XAS(calc2)
            x2, y2, ek2 = xas2.get_matrix_element(raw=True)

            mpi.send((x2, y2, ek2), 0, mpi.world)
            xas2.write('test.npz')
        elif rank == 0:
            calc1 = GPAW(gpw_files['si_corehole_nosym_pw'], communicator=comm)
            xas1 = XAS(calc1)

            x1, y1, ek1 = xas1.get_matrix_element(raw=True)

            x2, y2, ek2 = mpi.receive(1, mpi.world)
            print(ek1, ek2)
            assert ek2 == pytest.approx(ek1)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)
        assert 0

def test_si_xas_band_paralell(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    if mpi.size == 5:
        rank = mpi.world.rank

        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4])
        if rank in [1, 2, 3, 4]:
            parallel = {'band': 2}
            calc2 = GPAW(gpw_files['si_corehole_sym_pw'],
                         communicator=comm2,
                         parallel=parallel)
            xas2 = XAS(calc2)
            x2, y2, e_k2 = xas2.get_matrix_element(raw=True)

            mpi.send((x2, y2, e_k2), 0, mpi.world)
        elif rank == 0:
            calc1 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm)
            xas1 = XAS(calc1)

            x1, y1, e_k1 = xas1.get_matrix_element(raw=True)

            x2, y2, e_k2 = mpi.receive(1, mpi.world)

            assert e_k2 == pytest.approx(e_k1)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)


def test_si_xas_band_paralell_7(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    if mpi.size == 7:
        rank = mpi.world.rank

        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4, 5, 6])
        if rank in [1, 2, 3, 4, 5, 6]:
            parallel = {'band': 3}
            calc2 = GPAW(gpw_files['si_corehole_sym_pw'],
                         communicator=comm2,
                         parallel=parallel)
            xas2 = XAS(calc2)
            x2, y2, e_k2 = xas2.get_matrix_element(raw=True)

            mpi.send((x2, y2, e_k2), 0, mpi.world)
        elif rank == 0:
            calc1 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm)
            xas1 = XAS(calc1)

            x1, y1, e_k1 = xas1.get_matrix_element(raw=True)

            x2, y2, e_k2 = mpi.receive(1, mpi.world)

            assert e_k2 == pytest.approx(e_k1)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)


def test_si_xes_paralell(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    if mpi.size == 5:
        rank = mpi.world.rank

        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4])
        if rank in [1, 2, 3, 4]:
            calc2 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm2)
            xas2 = XAS(calc2, 'xes')
            x2, y2 = xas2.get_matrix_element()

            mpi.send((x2, y2), 0, mpi.world)
        elif rank == 0:
            calc1 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm)
            xas1 = XAS(calc1, 'xes')

            x1, y1 = xas1.get_matrix_element()

            x2, y2 = mpi.receive(1, mpi.world)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)


def test_si_all_paralell(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):

    if mpi.size == 5:
        rank = mpi.world.rank

        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4])
        if rank in [1, 2, 3, 4]:
            calc2 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm2)
            xas2 = XAS(calc2, 'all')
            x2, y2 = xas2.get_matrix_element()

            mpi.send((x2, y2), 0, mpi.world)
        elif rank == 0:
            calc1 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm)
            xas1 = XAS(calc1, 'all')

            x1, y1 = xas1.get_matrix_element()

            x2, y2 = mpi.receive(1, mpi.world)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)


def test_si_all_band_paralell_7(in_tmp_dir, add_cwd_to_setup_paths, gpw_files):
    if mpi.size == 7:
        rank = mpi.world.rank
        print(rank)
        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4, 5, 6])
        if rank in [1, 2, 3, 4, 5, 6]:
            parallel = {'band': 3}
            calc2 = GPAW(gpw_files['si_corehole_sym_pw'],
                         communicator=comm2,
                         parallel=parallel)
            xas2 = XAS(calc2, 'all')
            x2, y2, e_k2 = xas2.get_matrix_element(raw=True)

            mpi.send((x2, y2, e_k2), 0, mpi.world)
        elif rank == 0:
            calc1 = GPAW(gpw_files['si_corehole_sym_pw'], communicator=comm)
            xas1 = XAS(calc1, 'all')

            x1, y1, e_k1 = xas1.get_matrix_element(raw=True)

            x2, y2, e_k2 = mpi.receive(1, mpi.world)

            assert e_k2 == pytest.approx(e_k1)
            assert x2 == pytest.approx(x1)
            assert y2 == pytest.approx(y1)


def test_si_xas_parallel_xch(in_tmp_dir, add_cwd_to_setup_paths):
    setupname = 'Si1s1ch'
    gen('Si', name=setupname, corehole=(1, 0, 1), gpernode=30, write_xml=True)

    if mpi.world.size == 5:
        atoms = bulk('Si', 'diamond', 5.42)
        rank = mpi.world.rank
        comm = mpi.world.new_communicator([0])
        comm2 = mpi.world.new_communicator([1, 2, 3, 4])
        nbands = 10

        if rank in [1, 2, 3, 4]:

            calc2 = GPAW(mode='fd', kpts=[2, 2, 2], h=0.2,
                         nbands=-nbands,
                         spinpol=True,
                         setups={0: 'Si1s1ch'},
                         txt=None,
                         communicator=comm2,
                         occupations=FermiDirac(0.05, fixmagmom=True))

            atoms.calc = calc2

            atoms.get_potential_energy()
            xas2 = XAS(calc2, spin=0)

            x2, y2, e_k2 = xas2.get_matrix_element(raw=True)

            mpi.send((x2, y2, e_k2), 0, mpi.world)
        elif rank == 0:
            calc1 = GPAW(mode='fd', kpts=[2, 2, 2], h=0.2,
                         nbands=-nbands,
                         spinpol=True,
                         setups={0: 'Si1s1ch'},
                         txt=None,
                         communicator=comm,
                         occupations=FermiDirac(0.05, fixmagmom=True))
            atoms.calc = calc1
            atoms.get_potential_energy()

            xas1 = XAS(calc1, spin=0)
            x1, y1, e_k1 = xas1.get_matrix_element(raw=True)

            x2, y2, e_k2 = mpi.receive(1, mpi.world)

            assert e_k2 == pytest.approx(e_k1, 1.e-4)
            assert x2 == pytest.approx(x1, 1.e-2)
            assert y2 == pytest.approx(y1, abs=1.6e-5)
