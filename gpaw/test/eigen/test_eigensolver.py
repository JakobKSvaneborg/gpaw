import pytest
from ase import Atoms
from gpaw.core.matrix import suggest_blocking
from gpaw.dft import DFT
from gpaw.mpi import world


def parallelizations(size: int) -> list[tuple[int, int, int, int]]:
    """k-point, bands, domain, scalapack"""
    kbds = []
    for k in [1, 2]:
        for b in [1, 2]:
            for d in [1, 2, 4, 8]:
                if k * b * d != size:
                    continue
                for s in range(b * d + 1):
                    kbds.append((k, b, d, s))
    return kbds


@pytest.mark.parametrize('k, b, d, s', parallelizations(world.size))
def test_eigensolver(k, b, d, s):
    parallel = dict(
        kpt=k,
        band=b,
        domain=d)
    if s:
        parallel['sl_diagonalize'] = suggest_blocking(100, s)
    atoms = Atoms('H2', [[0, 0, 0], [0, 0, 0.75]], cell=[2, 2, 3], pbc=True)
    dft = DFT(
        atoms,
        mode='pw',
        eigensolver='davidson',
        kpts=(4, 1, 1),
        parallel=parallel)
    dft.converge(steps=3)
    e = dft.calculate_energy()
    assert e == pytest.approx(-11.893955)


if __name__ == '__main__':
    test_eigensolver(1, 1, 1, 0)
