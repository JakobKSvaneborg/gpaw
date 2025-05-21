import pytest
from ase import Atoms
from gpaw import GPAW, PW


@pytest.mark.hybrids
def test_h2(in_tmp_dir):
    L = 2.6
    a = Atoms('H2',
              [[0, 0, 0], [0.5, 0.5, 0]],
              cell=[L, L, 1],
              pbc=1)
    a.center()

    a.calc = GPAW(
        mode=PW(400, force_complex_dtype=True),
        symmetry='off',
        kpts={'size': (1, 1, 1), 'gamma': True},
        convergence={'density': 1e-6},
        # spinpol=True,
        txt='H2.txt',
        setups='ae',
        xc='HSE06'
        )
    e = a.get_potential_energy()
    eigs = a.calc.get_eigenvalues()
    print(e, eigs)


if __name__ == '__main__':
    test_h2(1)
    if 0:
        from cProfile import Profile
        prof = Profile()
        prof.enable()
        test_h2(1)
        prof.disable()
        from gpaw.mpi import rank, size
        prof.dump_stats(f'prof-{size}.{rank}')
