# creates: occupation_numbers.png
import numpy as np
import matplotlib.pyplot as plt

from gpaw.occupations import fermi_dirac, marzari_vanderbilt, methfessel_paxton

width = 0.2
x = np.linspace(-0.7, 0.7, 101)

ax = plt.subplot()
ax.plot(x, fermi_dirac(x, 0.0, width / 4)[0],
        label=f'fermi-dirac ({width / 4:.2f} eV)')
ax.plot(x, marzari_vanderbilt(x, 0.0, width)[0],
        label=f'marzari-vanderbilt ({width:.1f} eV)')
ax.plot(x, methfessel_paxton(x, 0.0, width, 0)[0],
        label=f'methfessel_paxton-0 ({width:.1f} eV)')
ax.plot(x, methfessel_paxton(x, 0.0, width, 1)[0],
        label=f'methfessel_paxton-1 ({width:.1f} eV)')
ax.axvline(0.0)
plt.xlabel('energy [eV]')
plt.ylabel('occupation')
plt.legend()
# plt.show()
plt.savefig('occupation_numbers.png')
