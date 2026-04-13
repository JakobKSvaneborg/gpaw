import numpy as np
from ase import Atoms

from gpaw import GPAW, GPAW_NEW, PW, RMMDIIS
from gpaw.hybrids import HybridXC
from gpaw.mpi import world


def run(symbol, d0, M, ecut, L):
    a = Atoms(2 * symbol, cell=[L, L, L])
    a.positions[1, 0] = d0
    a.center()
    D = np.linspace(d0 * 0.98, d0 * 1.02, 5)
    E = []
    for d in D:
        a.set_distance(0, 1, d)
        a.calc = GPAW(
            mode=PW(ecut),
            xc={'name': 'PBE0', 'backend': 'pw'},
            nbands=0,
            # eigensolver='rmm-diis',
            txt=f'{symbol}2-{d / d0:.2f}-{ecut}-{L}.txt')
        e = a.get_potential_energy()
        E.append(e)

    p0 = np.polyfit(D, E, 3)
    p1 = np.polyder(p0)
    p2 = np.polyder(p1)
    d = sorted([(np.polyval(p2, d), d) for d in np.roots(p1)])[1][1]
    e2 = np.polyval(p0, d)

    a = Atoms(symbol, cell=[L, L, L], magmoms=[M])
    a.center()
    a.calc = GPAW(
        mode=PW(ecut),
        txt=f'{symbol}-{ecut}-{L}.txt',
        xc={'name': 'PBE0', 'backend': 'pw'},
        **(dict() if GPAW_NEW else
           dict(eigensolver=RMMDIIS(niter=1),
                parallel={'band': 1, 'kpt': 1})))
    e1 = a.get_potential_energy()

    if world.rank == 0:
        print(symbol, ecut, L, 2 * e1 - e2, d)

    return 2 * e1 - e2, d


if __name__ == '__main__':
    for L in [8, 10, 12]:
        run('H', 0.75, 1, 500, L)
        run('N', 1.089, 3, 500, L)
    for ecut in [400, 500, 600, 700]:
        run('H', 0.75, 1, ecut, 8)
        run('N', 1.089, 3, ecut, 8)
