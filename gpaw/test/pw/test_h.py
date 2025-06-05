from ase.build import molecule
from gpaw import GPAW, PW
from gpaw.mpi import world
from gpaw.new.pwfd.pcg import NotDavidson


def test_pw_h(in_tmp_dir):
    a = molecule('H2', pbc=1)
    a.center(vacuum=2)

    #comm = world.new_communicator([world.rank])
    #e0 = 0.0
    #a.calc = GPAW(mode=PW(250),
    #              communicator=comm,
    #              txt=None)
    #e0 = a.get_potential_energy()
    #e0 = world.sum_scalar(e0) / world.size

    a.calc = GPAW(mode=PW(250),
                  random=True,
                  eigensolver={'name': 'not-dav',
                               'niter': 10},
                  basis='szp(dzp)',
                  convergence={'eigenvalues': 1e-4},)
                  #txt='%d.txt' % world.size)
    e = a.get_potential_energy()
    f = a.get_forces()
    assert abs(e - e0) < 7e-5, abs(e - e0)
    assert abs(f).max() < 1e-10, abs(f).max()
