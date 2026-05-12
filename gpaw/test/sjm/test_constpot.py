import pytest


@pytest.mark.skip('https://gitlab.com/gpaw/gpaw/-/issues/1381')
def test_constpot(atoms):
    atoms.calc.set(sj={'tol': 0.5})
    atoms.get_forces()

    pot = atoms.calc.parameters['sj']['target_potential']
    tol = atoms.calc.parameters['sj']['tol']
    assert abs(atoms.calc.get_electrode_potential() - pot) < tol
