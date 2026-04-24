import pytest
from ase.build import molecule
from ase.units import kcal, mol

from gpaw import GPAW
from gpaw.solvation import SolvationGPAW, get_HW14_water_kwargs
from gpaw.utilities.adjust_cell import adjust_cell


parameters = {
    'mode': 'fd',
    'xc': 'PBE',
    'h': 0.24,
    'convergence': {
        'energy': 0.05 / 8.,
        'density': 10.,
        'eigenstates': 10.}}


def h2o(legacy_gpaw):
    vac = 4.0

    atoms = molecule('H2O')
    adjust_cell(atoms, vac, parameters['h'])

    kwargs = get_HW14_water_kwargs()
    kwargs.update(parameters)
    atoms.calc = SolvationGPAW(**kwargs, legacy_gpaw=legacy_gpaw)
    atoms.get_potential_energy()

    return atoms


@pytest.mark.parametrize('legacy', [True, False])
def test_solvation_water_water(legacy):
    SKIP_VAC_CALC = True

    H2O = h2o(legacy)

    if not SKIP_VAC_CALC:
        atoms = H2O.copy()
        atoms.calc = GPAW(**parameters)
        Evac = atoms.get_potential_energy()
    else:
        # h=0.24, vac=4.0, setups: 0.9.11271, convergence: only energy 0.05 / 8
        Evac = -14.857414548

    Ewater = H2O.get_potential_energy()
    H2O.get_forces()
    DGSol = (Ewater - Evac) / (kcal / mol)
    print('Delta Gsol: %s kcal / mol' % DGSol)

    assert DGSol == pytest.approx(-6.3, abs=2.)

    if legacy:
        Eelwater = H2O.calc.get_electrostatic_energy()
        Esurfwater = H2O.calc.get_solvation_interaction_energy('surf')
        assert Ewater == pytest.approx(Eelwater + Esurfwater, abs=1e-14)
    else:
        Esurfwater = H2O.calc.dft.solvation.interaction_energy()
    assert Esurfwater == pytest.approx(0.058, abs=0.002)


@pytest.mark.filterwarnings('ignore:unclosed file')
@pytest.mark.parametrize('legacy', [True, False])
def test_read(in_tmp_dir, legacy):
    """Read and check some basic properties"""
    H2O = h2o(legacy)
    fname = 'solvation.gpw'
    H2O.calc.write(fname)

    if legacy:
        calc = SolvationGPAW(fname)
    else:
        calc = GPAW(fname)

    for method in ['get_potential_energy',
                   'get_eigenvalues', 'get_occupation_numbers']:
        assert getattr(calc, method)() == pytest.approx(
            getattr(H2O.calc, method)())

    calc.get_atoms().get_potential_energy()

    for method in ['get_potential_energy',
                   'get_eigenvalues', 'get_occupation_numbers']:
        assert getattr(calc, method)() == pytest.approx(
            getattr(H2O.calc, method)())
