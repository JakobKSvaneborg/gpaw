import pytest
from .base_calc import calculator


@pytest.mark.old_gpaw_only
# @pytest.mark.ci maybe
def test_dirichlet(atoms):
    atoms.calc.set(sj={'tol': 0.01})
    E1 = atoms.get_potential_energy()
    pot = atoms.calc.parameters['sj']['target_potential']
    tol = atoms.calc.parameters['sj']['tol']

    atoms.calc = calculator()
    atoms.calc.set(sj={'tol': 0.01, 'dirichlet': True})
    E2 = atoms.get_potential_energy()

    assert abs(atoms.calc.get_electrode_potential() - pot) < tol
    assert abs(-atoms.calc.get_fermi_level() - pot) < tol
    assert abs(E1 - E2) < 1e-4
