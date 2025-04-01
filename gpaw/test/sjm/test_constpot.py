import pytest
from .base_calc import atoms, calculator


@pytest.mark.old_gpaw_only
# @pytest.mark.ci maybe
def test_constpot():
    calc = calculator()

    atoms.calc = calc
    atoms.calc.set(sj={'tol': 0.5})

    atoms.get_forces()

    pot = atoms.calc.parameters['sj']['target_potential']
    tol = atoms.calc.parameters['sj']['tol']
    assert abs(atoms.calc.get_electrode_potential() - pot) < tol


if __name__ == '__main__':
    test_constpot()
