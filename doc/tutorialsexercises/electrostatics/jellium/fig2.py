# web-page: fig2.png
import numpy as np
import matplotlib.pyplot as plt
from ase.units import Bohr
from gpaw import GPAW
rs = 5.0 * Bohr
calc = GPAW('surface.gpw')
density = calc.get_pseudo_density()[0, 0]
h = 0.2
a = 8 * h
v = 3 * a
L = 10 * a
z = np.linspace(0, v + L + v, len(density), endpoint=False)
# Position of surface is between two grid points:
z0 = (v + L - h / 2)
n = 1 / (4 * np.pi / 3 * rs**3)  # electron density
kF = (3 * np.pi**2 * n)**(1.0 / 3)
lambdaF = 2 * np.pi / kF  # Fermi wavelength
plt.figure(figsize=(6, 6 / 2**0.5))
plt.plot([-L / lambdaF, -L / lambdaF, 0, 0], [0, 1, 1, 0], 'k')
plt.plot((z - z0) / lambdaF, density / n)
# plt.xlim(xmin=-1.2, xmax=1)
plt.ylim(ymin=0)
plt.title(rf'$r_s={rs / Bohr:.1f}\ \mathrm{{Bohr}},\ '
          rf'\lambda_F={lambdaF / Bohr:.1f}\ \mathrm{{Bohr}}$')
plt.xlabel('DISTANCE (FERMI WAVELENGTHS)')
plt.ylabel('ELECTRON DENSITY')
plt.savefig('fig2.png')
