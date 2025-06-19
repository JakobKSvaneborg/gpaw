from gpaw import GPAW
from gpaw.borncharges import born_charges_wf
from ase.build import mx2

calc_params = {
    'mode': {'name': 'lcao'},
    'xc': 'PBE',
    'basis': 'dzp',
    'kpts': {'density': 3.0},
    'occupations': {'name': 'fermi-dirac', 'width': 0.05},
    'symmetry': 'off',
    'convergence': {'density': 1e-4},
    'txt': 'test.txt'
}

atoms = mx2('MoS2', vacuum=5.0)
atoms.center()

calc = GPAW(**calc_params)
atoms.calc = calc
results = born_charges_wf(atoms)
