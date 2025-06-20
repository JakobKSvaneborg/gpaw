from gpaw import GPAW
from gpaw.borncharges import born_charges_wf
from ase.build import mx2
from pathlib import Path

calc_params = {
    'mode': {'name': 'pw', 'ecut': 400},
    'xc': 'PBE',
    'kpts': {'density': 3.0},
    'occupations': {'name': 'fermi-dirac', 'width': 0.05},
#    'symmetry': 'off',
    'convergence': {'density': 1e-4},
}

atoms = mx2('MoS2', vacuum=5.0)
atoms.center()

gpw_file = Path('MoS2.gpw')

calc = GPAW(**calc_params, txt='gs.txt')
atoms.calc = calc
atoms.get_potential_energy()
atoms.calc.write(gpw_file, mode='all')

#results = born_charges_wf(atoms, cleanup=True)
results = born_charges_wf(atoms, gpw_file=gpw_file)
