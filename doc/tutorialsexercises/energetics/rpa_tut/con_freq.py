from ase.parallel import paropen
from ase.units import Hartree

from gpaw.xc.rpa import RPACorrelation

f = paropen('con_freq.dat', 'w')
for N in [4, 6, 8, 12, 16, 24, 32]:
    rpa = RPACorrelation('N2.gpw', txt='rpa_N2_frequencies.txt',
                         ecut=[50],
                         nfrequencies=N)
    data = rpa.calculate_all_contributions()
    energy = data.energy_i[0] * Hartree
    print(N, energy, file=f)
    if N == 16:
        f16 = paropen('frequency_gauss16.dat', 'w')
        for omega, energy_i in zip(
                data.integral.omega_w, data.energy_wi):
            print(omega * Hartree, energy_i[0], file=f16)
        f16.close()
f.close()
