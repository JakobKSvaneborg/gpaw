import pytest
from .base_calc import calculator
from ase.build import fcc111


@pytest.mark.old_gpaw_only
@pytest.mark.ci
@pytest.mark.serial
def test_keys():
    atoms = fcc111('H', size=(1, 1, 1), a=2.5)
    atoms.center(axis=2, vacuum=5)
    atoms.cell[2][2] = 10

    calc = calculator()
    calc.set(sj={'excess_electrons': 1.,
                 'jelliumregion': {'top': -2.,
                                   'bottom': -4.,
                                   'thickness': None,
                                   'fix_bottom': False},
                 'target_potential': 3,
                 'pot_ref': 'wf',
                 'tol': 0.01,
                 'always_adjust': True,
                 'grand_output': False,
                 'max_iters': 100,
                 'max_step': 3.,
                 'slope': 5,
                 'mixer': 1,
                 'fdt': False,
                 'slope_regression_depth': 4,
                 'dirichlet': False,
                 'cip': {'autoinner': {'nlayers': None,
                                       'threshold': 0.0001},
                         'inner_region': None,
                         'mu_pzc': None,
                         'phi_pzc': None,
                         'filter': 10}})
    atoms.calc = calc
    atoms.calc.initialize(atoms)
    atoms.calc.initialize_positions(atoms)


if __name__ == '__main__':
    test_keys()
