import pytest
from .base_calc import calculator
from ase.build import fcc111


@pytest.mark.old_gpaw_only
def test_cavitylike():
    atoms = fcc111('H', size=(1, 1, 1), a=2.5)
    atoms.center(axis=2, vacuum=5)
    atoms.cell[2][2] = 10

    charge = 0.2
    calc = calculator()
    calc.set(sj={'jelliumregion': {'bottom': 'cavity_like'},
                 'target_potential': None,
                 'excess_electrons': 0.2})
    atoms.calc = calc
    atoms.get_potential_energy()
    bc = atoms.calc.parameters.background_charge.todict()
    assert bc['charge'] == charge
    assert bc['z1'] == 'cavity_like'


if __name__ == '__main__':
    test_cavitylike()
