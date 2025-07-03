import pytest
from ase import Atoms
from gpaw import GPAW, PW


@pytest.mark.libxc
@pytest.mark.hybrids
@pytest.mark.new_gpaw_ready
def test_exx_double_cell(in_tmp_dir, gpaw_new):
    L = 2.6
    a = Atoms('H2',
              [[0, 0, 0], [0.5, 0.5, 0]],
              cell=[L, L, 1],
              pbc=1)
    a.center()

    kwargs = dict(
        mode=PW(400),
        symmetry='off',
        convergence={'density': 1e-6},
        spinpol=True,
        xc='HSE06')
    if gpaw_new:
        kwargs |= dict(spinpol=False,
                       # setups='ae',
                       eigensolver='rmm-diis')

    a.calc = GPAW(
        kpts={'size': (1, 1, 4), 'gamma': True},
        txt='H2-new.txt',
        **kwargs)
    e1 = a.get_potential_energy()
    eps1 = a.calc.get_eigenvalues(1)[0]
    if not gpaw_new:
        f1 = a.get_forces()
        assert abs(f1[1, 0] - 9.60644) < 0.0005
    if 0:
        # To check against numeric calculation of the forces, but it takes
        # more time
        from gpaw.test import calculate_numerical_forces
        f1n = calculate_numerical_forces(a, 0.001, [1], [0])[0, 0]
        assert abs(f1[1, 0] - f1n) < 0.0005

    a *= (1, 1, 2)
    a.calc = GPAW(
        kpts={'size': (1, 1, 2), 'gamma': True},
        txt='H4-new.txt',
        **kwargs)
    e2 = a.get_potential_energy()
    eps2 = a.calc.get_eigenvalues(0)[0]
    if not gpaw_new:
        f2 = a.get_forces()

        f2[:2] -= f1
        f2[2:] -= f1
        assert abs(f2).max() < 0.00085

    assert abs(e2 - 2 * e1) < 0.002
    assert abs(eps1 - eps2) < 0.001


if __name__ == '__main__':
    if 0:
        from cProfile import Profile
        prof = Profile()
        prof.enable()
        test_exx_double_cell(1)
        prof.disable()
        from gpaw.mpi import rank, size
        prof.dump_stats(f'prof-{size}.{rank}')
    else:
        test_exx_double_cell(1, 1)
