import sys
from ase.build import bulk
from gpaw import GPAW
from pathlib import Path

# Script to get the total energies of a supercell
# of GaAs with and without a Ga vacancy

N = int(sys.argv[1])  # NxNxN supercell
label = f'GaAs_{N}x{N}x{N}'
prs_path = Path(f'{label}_prs.gpw')
def_path = Path(f'{label}_def.gpw')

a0 = 5.628      # lattice parameter
charge = -3     # defect charge

params = {'mode': {'name': 'pw', 'ecut': 400},
          'xc': 'LDA',
          'kpts': {'size': (2, 2, 2), 'gamma': False},
          'occupations': {'name': 'fermi-dirac', 'width': 0.01}}

calc_charged = GPAW(charge=charge, **params)
calc_neutral = GPAW(charge=0, **params)

prim = bulk('GaAs', crystalstructure='zincblende', a=a0, cubic=True)
pristine = prim * (N, N, N)
pristine.calc = calc_neutral
pristine.get_potential_energy()
pristine.calc.write(prs_path)

defect = pristine.copy()
defect.pop(0)  # make a Ga vacancy
defect.calc = calc_charged
defect.get_potential_energy()
defect.calc.write(def_path)
