import pytest


@pytest.mark.old_gpaw_only
# @pytest.mark.ci maybe
def test_dirichlet(atoms):
    atoms.calc.set(sj={'tol': 0.01})
    E1 = atoms.get_potential_energy()
    pot = atoms.calc.parameters['sj']['target_potential']
    tol = atoms.calc.parameters['sj']['tol']

    atoms.calc.results={}
    #atoms.calc.density.reset()
    atoms.calc.scf.reset()
    atoms.calc.set(sj={'dirichlet': True})
    E2 = atoms.get_potential_energy()
    print(E1, E2, pot, -atoms.calc.get_fermi_level())

#    assert abs(atoms.calc.get_electrode_potential() - pot) < tol
    assert abs(-atoms.calc.get_fermi_level() - pot) < tol
    assert abs(E1 - E2) < 1e-4
