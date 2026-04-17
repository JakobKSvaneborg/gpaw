import pytest
from ase import Atoms
from gpaw.dft import GPAW, PW
from gpaw.mpi import world


def par(size):
    kb = []
    for k in range(1, 3):
        if size % k != 0:
            continue
        for b in range(1, size // k + 1):
            if (size // k) % b == 0:
                kb.append((k, b))
    return kb


@pytest.mark.parametrize(
    'kb',
    [pytest.param((k, b), id=f'k{k}b{b}') for k, b in par(world.size)])
def test_all(kb):
    k, b = kb
    L = 2.6
    a = Atoms('Li2',
              [[0, 0, 0], [0.9, 0.9, 0]],
              cell=[L, L, 1.4],
              pbc=1)
    a.center()

    kwargs = dict(
        mode=PW(400),
        convergence={'density': 1e-7},
        mixer={'beta': 0.25},
        xc='HSE06')
    a.calc = GPAW(
        kpts={'size': (1, 1, 4), 'gamma': True},
        #txt=f'Li2-{world.size}.txt',
        parallel={'kpt': k, 'band': b},
        **kwargs)
    e1 = a.get_potential_energy()
    print(e1)
    assert e1 == pytest.approx(-2.6074285563393125)
    f1 = a.get_forces()
    print(f1)
    assert f1[0, 0] == pytest.approx(-1.44417016, abs=5e-6)
    assert f1[0, 0] == pytest.approx(f1[0, 1])
    assert f1[0, 0] == pytest.approx(-f1[1, 0])
    assert f1[0, 0] == pytest.approx(-f1[1, 1])


if __name__ == '__main__':
    import sys
    test_all([int(x) for x in sys.argv[1:]])
    # print(par(2))
