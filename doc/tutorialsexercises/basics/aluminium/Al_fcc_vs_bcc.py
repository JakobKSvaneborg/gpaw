"""Compare Al(fcc) and Al(bcc) at two different plane-wave cutoffs
and two differens k-point densities."""
from ase.build import bulk
from gpaw import GPAW, PW

afcc = 3.985
abcc = 3.190

for kdens in [2.0, 3.0]:
    for ecut in [300, 500]:
        fcc = bulk('Al', 'fcc', a=afcc)
        calc = GPAW(mode=PW(ecut),
                    kpts={'density': kdens},
                    txt=f'bulk-fcc-{ecut:.1f}-{kdens:.1f}.txt')
        fcc.calc = calc
        efcc = fcc.get_potential_energy()

        bcc = bulk('Al', 'bcc', a=abcc)
        calc = GPAW(mode=PW(ecut),
                    kpts={'density': 4.0},
                    txt=f'bulk-bcc-{ecut:.1f}-{kdens:.1f}.txt')
        bcc.calc = calc
        ebcc = bcc.get_potential_energy()

        print(kdens, ecut, efcc, ebcc, efcc - ebcc)
