from ase.parallel import paropen
from ase.units import Hartree
from gpaw.xc.rpa import RPACorrelation

rpa = RPACorrelation('N.gpw', nblocks=8,
                     txt='rpa_N.txt', ecut=400)
E1_i = rpa.calculate()

rpa = RPACorrelation('N2.gpw', nblocks=8,
                     txt='rpa_N2.txt', ecut=400)
E2_i = rpa.calculate()
ecut_i = rpa.integral.ecut_i

f = paropen('rpa_N2.dat', 'w')
for ecut, E1, E2 in zip(ecut_i, E1_i, E2_i):
    print(ecut * Hartree, E2 - 2 * E1, file=f)
f.close()
