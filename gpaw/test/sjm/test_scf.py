import pytest
from .base_calc import calculator
from ase.build import fcc111


@pytest.mark.old_gpaw_only
@pytest.mark.ci
def test_scf():
    atoms = fcc111('H', size=(1, 1, 1), a=2.5)
    atoms.center(axis=2, vacuum=5)
    atoms.cell[2][2] = 10

    calc = calculator()
    calc.set(sj={'target_potential': 3.64,
                 'excess_electrons': 0.02,
                 'tol': 0.5})
    atoms.calc = calc
    atoms.get_potential_energy()


if __name__ == '__main__':
    test_scf()
