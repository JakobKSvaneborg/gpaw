import pytest
from .base_calc import calculator
from ase.build import fcc111


@pytest.mark.old_gpaw_only
def test_change_potential():
    atoms = fcc111('H', size=(1, 1, 1), a=2.5)
    atoms.center(axis=2, vacuum=5)
    atoms.cell[2][2] = 10

    atoms.calc = calculator()
    atoms.calc.set(sj={'tol': 0.1})

    pot = atoms.calc.parameters['sj']['target_potential']
    tol = atoms.calc.parameters['sj']['tol']
    E1 = atoms.get_potential_energy()
    print(pot, atoms.calc.get_electrode_potential())
    assert abs(atoms.calc.get_electrode_potential() - pot) < tol

    atoms.calc.set(sj={'target_potential': pot - 0.1})
    atoms.get_potential_energy()
    assert abs(atoms.calc.get_electrode_potential() - pot + 0.1) < tol

    atoms.calc.set(sj={'target_potential': pot})
    E2 = atoms.get_potential_energy()
    assert abs(atoms.calc.get_electrode_potential() - pot) < tol
    assert abs(E1 - E2) < 1e-2


if __name__ == '__main__':
    test_change_potential()
