import pytest


@pytest.mark.skip('See #1510')
# @pytest.mark.ci maybe
def test_dirichlet(atoms):
    atoms.calc.set(sj={'tol': 0.5, 'dirichlet': True})
    atoms.get_forces()

    pot = atoms.calc.parameters['sj']['target_potential']
    tol = atoms.calc.parameters['sj']['tol']
    assert abs(atoms.calc.get_electrode_potential() - pot) < tol
    assert abs(-atoms.calc.get_fermi_level() - pot) < tol
