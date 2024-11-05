import pytest
from ase import Atoms
from gpaw import GPAW


def test_scissors():
    """Opend gap in one of two isolated H2 moleculs."""
    h2 = Atoms('2H2', [[0, 0, 0], [0, 0, 0.74],
                       [4, 0, 0], [4, 0, 0.74]])
    h2.center(vacuum=3.0)
    d = 1.0
    h2.calc = GPAW(mode='lcao',
                   basis='sz(dzp)',
                   eigensolver={'name': 'scissors',
                                'shifts': [(-d, d, 2)]},
                   txt=None)
    h2.get_potential_energy()
    eigs1 = h2.calc.get_eigenvalues()
    i, ii, iii, iv = eigs1
    assert ii - i == pytest.approx(d, abs=0.01)
    assert iv - iii == pytest.approx(d, abs=0.01)

    # Check also fixed-density calculations:
    eigs2 = h2.calc.fixed_density(kpts=[[0, 0, 0]]).get_eigenvalues()
    assert eigs2 == pytest.approx(eigs1)
