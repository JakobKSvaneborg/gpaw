import sys
import matplotlib.pyplot as plt
from gpaw.dos import DOSCalculator

# The following five lines read a file name and an optional width
# from the command line.
filename = sys.argv[1]
if len(sys.argv) > 2:
    width = float(sys.argv[2])
else:
    width = 0.1

dos = DOSCalculator.from_calculator(filename)
energies = dos.get_energies()
if dos.nspins == 2:
    plt.plot(energies, dos.raw_dos(energies, spin=0, width=width))
    plt.plot(energies, dos.raw_dos(energies, spin=1, width=width))
    plt.legend(('up', 'down'), loc='upper left')
else:
    plt.plot(energies, dos.raw_dos(energies, width=width))
plt.xlabel(r'$\epsilon - \epsilon_F \ \rm{(eV)}$')
plt.ylabel('Density of States (1/eV)')
plt.show()
