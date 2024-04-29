from ase.build import mx2
from gpaw.new.ase_interface import GPAW
import numpy as np

atoms = mx2('NiI2', kind='1T', a=3.969662131560825,
            thickness=3.027146598949815, vacuum=4)

# Align the magnetic moment in the xy-plane
magmoms = [[1, 0, 0], [0, 0, 0], [0, 0, 0]]
ecut = 600
k = 6

# Construct list of q-vectors
path = atoms.cell.bandpath('GMKG', npoints=31).kpts

energies_q = []
magmoms_q = []

for i, q_c in enumerate(path):
    # Spin-spiral calculations require non-collinear calculations
    # without symmetry or spin-orbit coupling
    calc = GPAW(mode={'name': 'pw',
                      'ecut': ecut,
                      'qspiral': q_c},
                xc='LDA',
                mixer={'backend': 'pulay',
                       'beta': 0.05,
                       'method': 'sum',
                       'nmaxold': 5,
                       'weight': 100},
                symmetry='off',
                parallel={'domain': 1, 'band': 1},
                magmoms=magmoms,
                kpts={'density': 6.0, 'gamma': True},
                txt=f'gsq-{i:02}.txt')
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    calc.write(f'gsq-{i:02}.gpw')
    magmom = atoms.calc.dft.magmoms()[0]
    energies_q.append(energy)
    magmoms_q.append(np.linalg.norm(magmom))

energies_q = np.array(energies_q)
magmoms_q = np.array(magmoms_q)
np.savez('data.npz', energies=energies_q, magmoms=magmoms_q)
