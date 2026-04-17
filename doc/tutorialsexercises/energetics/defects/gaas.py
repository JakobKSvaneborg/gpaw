"""Script to get the total energies of a supercell
of GaAs with and without a Ga vacancy
"""
import sys
from ase.build import bulk
from gpaw import GPAW

a0 = 5.628      # lattice parameter

N = int(sys.argv[1])  # NxNxN supercell
charge = int(sys.argv[2])
label = f'GaAs_{N}x{N}x{N}'
tag = 'def' if charge == 0 else 'prs'
params = {'mode': {'name': 'pw', 'ecut': 400},
          'xc': 'LDA',
          'kpts': {'size': (2, 2, 2), 'gamma': False},
          'occupations': {'name': 'fermi-dirac', 'width': 0.01},
          'txt': f'{label}_{tag}.txt'}
prim = bulk('GaAs', crystalstructure='zincblende', a=a0, cubic=True)
atoms = prim * (N, N, N)
if charge == -3:
    atoms.pop(0)  # make a Ga vacancy
atoms.calc = GPAW(charge=charge, **params)
atoms.get_potential_energy()
atoms.calc.write(f'{label}_{tag}.gpw')
