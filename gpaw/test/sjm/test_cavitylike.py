import pytest
from .base_calc import atoms, calculator


@pytest.mark.old_gpaw_only
def test_cavitylike():
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
