import pytest
from ase import Atoms
from gpaw import GPAW, PW


@pytest.mark.hybrids
def test_h2(in_tmp_dir, c=False):
    L = 2.6
    a = Atoms('H2',
              [[0, 0, 0], [0.5, 0.5, 0]],
              cell=[L, L, 1],
              pbc=1)
    a.center()

    a.calc = GPAW(
        mode=PW(400, force_complex_dtype=c),
        symmetry='off',
        kpts={'size': (1, 1, 2)},  # 'gamma': not True},
        convergence={'density': 1e-6},
        eigensolver={'name': 'dav', 'niter': 1},
        nbands=1,
        # spinpol=True,
        # txt='H2.txt',
        setups='ae',
        xc='HSE06')
    e = a.get_potential_energy()
    eigs = a.calc.get_eigenvalues()
    print(e, eigs)


if __name__ == '__main__':
    import sys
    test_h2(1, bool(int(sys.argv[1])))
    if 0:
        from cProfile import Profile
        prof = Profile()
        prof.enable()
        test_h2(1)
        prof.disable()
        from gpaw.mpi import rank, size
        prof.dump_stats(f'prof-{size}.{rank}')
