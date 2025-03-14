from ase.parallel import paropen
from ase.units import Hartree
from gpaw.xc.rpa import RPACorrelation
from gpaw.xc.fxc import FXCCorrelation

fxc = FXCCorrelation('diamond.ralda.pbe_wfcs.gpw', xc='rAPBE',
                     ecut=400,
                     nblocks=8,
                     txt='diamond.ralda_02_rapbe.txt')
E_i = fxc.calculate()

f = paropen('diamond.ralda.rapbe.dat', 'w')
for ecut, E in zip(fxc.rpa.integral.ecut_i, E_i):
    print(ecut * Hartree, E, file=f)
f.close()

rpa = RPACorrelation('diamond.ralda.pbe_wfcs.gpw',
                     ecut=400,
                     nblocks=8,
                     txt='diamond.ralda_02_rpa.txt')
E_i = rpa.calculate()

f = paropen('diamond.ralda.rpa.dat', 'w')
for ecut, E in zip(rpa.integral.ecut_i, E_i):
    print(ecut * Hartree, E, file=f)
f.close()
