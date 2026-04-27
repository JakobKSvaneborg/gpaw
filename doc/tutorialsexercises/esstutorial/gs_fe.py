from ase import Atoms
from gpaw import GPAW, PW

# bcc unit cell
unit_cell = [[-1.4335, 1.4335, 1.4335],
             [1.4335, -1.4335, 1.4335],
             [1.4335, 1.4335, -1.4335]]
# List of atoms
bulk = Atoms('Fe',
             scaled_positions=[[0, 0, 0]],
             cell=unit_cell,
             pbc=True)
bulk.set_initial_magnetic_moments([2.0])

Nk = 8
ecut = 600
calc = GPAW(mode=PW(ecut),
            xc='LDA',
            kpts=(Nk, Nk, Nk),
            # txt='gs_fe.txt',
            )

bulk.calc = calc
bulk.get_potential_energy()
calc.write('gs_fe.gpw')
