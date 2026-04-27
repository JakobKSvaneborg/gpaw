import matplotlib.pyplot as plt
from gpaw.dos import DOSCalculator

dos = DOSCalculator.from_calculator('ferro.gpw')
energies = dos.get_energies(emax=5.0)
# Plot s, p, d projected LDOS:
width = 0.4
for l, c in enumerate('spd'):
    for spin in [0, 1]:
        d = dos.raw_pdos(energies, a=0, l=l, spin=spin, width=width)
        plt.plot(energies, d, label=c + ('-up' if spin == 0 else '-down'))
plt.legend()
plt.show()
