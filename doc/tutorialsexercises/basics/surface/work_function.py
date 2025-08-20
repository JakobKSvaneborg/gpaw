import numpy as np
import matplotlib.pyplot as plt

from gpaw import GPAW

# Read in the 5-layer slab:
calc = GPAW('slab-5.gpw')
slab = calc.get_atoms()

# Get the height of the unit cell:
L = slab.get_cell()[2, 2]

# Get the effective potential on a 3D grid:
v = calc.get_effective_potential()

nx, ny, nz = v.shape
z = np.linspace(0, L, nz, endpoint=False)

efermi = calc.get_fermi_level()

# Calculate xy averaged potential:
vz = v.mean(axis=0).mean(axis=0)
print(f'Work function: {vz.max() - efermi:.2f} eV')

plt.plot(z, vz, label='xy averaged effective potential')
plt.plot([0, L], [efermi, efermi], label='Fermi level')
plt.ylabel('Potential / V')
plt.xlabel('z / Angstrom')
plt.legend(loc=0)
# plt.savefig('workfunction.png', format='png')
plt.show()
