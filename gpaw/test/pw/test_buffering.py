from gpaw import GPAW
from gpaw.mpi import world
from ase.build import molecule
import numpy as np
import memray
with memray.Tracker(f"output_rank_{world.rank}.bin"): 
      atoms = molecule('C60')
      atoms.center(vacuum=6)
      calc = GPAW(mode={'name': 'pw',
                        'ecut': 800,
                        'dtype': np.complex128},
                  random=True,
                  eigensolver='dav',
                  poissonsolver={'fast': 'fast'},
                  convergence={'maximum iterations': 2},
                  parallel={'band': 1})
      atoms.calc = calc
      atoms.get_potential_energy()