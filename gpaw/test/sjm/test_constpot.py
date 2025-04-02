import pytest
from .base_calc import calculator
from ase.build import fcc111


@pytest.mark.old_gpaw_only
# @pytest.mark.ci maybe
def test_constpot():
    atoms = fcc111('H', size=(1, 1, 1), a=2.5)
    atoms.center(axis=2, vacuum=5)
    atoms.cell[2][2] = 10

    calc = calculator()

    atoms.calc = calc
    atoms.calc.set(sj={'tol': 0.5})

    atoms.get_forces()

    pot = atoms.calc.parameters['sj']['target_potential']
    tol = atoms.calc.parameters['sj']['tol']
    assert abs(atoms.calc.get_electrode_potential() - pot) < tol


if __name__ == '__main__':
    test_constpot()
