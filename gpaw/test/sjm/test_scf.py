import pytest
from .base_calc import atoms, calculator


@pytest.mark.old_gpaw_only
@pytest.mark.ci
def test_scf():
    calc = calculator()
    calc.set(sj={'target_potential': 3.64,
                 'excess_electrons': 0.02,
                 'tol': 0.5})
    atoms.calc = calc
    atoms.get_potential_energy()


if __name__ == '__main__':
    test_scf()
