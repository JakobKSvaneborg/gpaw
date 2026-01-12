# web-page: au-ddos.png
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from gpaw import GPAW, GPAW_NEW

calc = GPAW('au.gpw')

if GPAW_NEW:
    dos = calc.dos()
    energies = dos.get_energies()
    pdos = dos.raw_pdos(energies, a=0, l=2)
else:
    energies, pdos = calc.get_orbital_ldos(a=0, angular='d')
    energies -= calc.get_fermi_level()

I = trapezoid(pdos, energies)
center = trapezoid(pdos * energies, energies) / I
width = np.sqrt(trapezoid(pdos * (energies - center)**2, energies) / I)
plt.plot(energies, pdos)
plt.xlabel('Energy (eV)')
plt.ylabel('d-projected DOS on atom 0')
plt.title(f'd-band center = {center:.1f} eV, d-band width = {width:.1f} eV')
# plt.show()
plt.savefig('au-ddos.png')
