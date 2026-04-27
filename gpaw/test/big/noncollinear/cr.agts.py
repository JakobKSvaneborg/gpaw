import numpy as np
from ase import Atoms

from gpaw import GPAW, PW, MixerDif


def workflow():
    from myqueue.workflow import run
    run(function=test, cores=8)


def angle(a, b):
    x = np.dot(a, b) / (np.dot(a, a) * np.dot(b, b))**0.5
    return np.arccos(x) * 180 / np.pi


def check(m_av):
    for a in range(3):
        theta = angle(m_av[a], m_av[a - 1])
        print(theta)
        assert abs(theta - 120) < 0.5, theta


def test():
    d = 2.66
    atoms = Atoms('Cr3', positions=[(0, 0, 0), (d, 0, 0), (2 * d, 0, 0)],
                  cell=[[d * 3 / 2, -d * np.sqrt(3) / 2, 0],
                        [d * 3 / 2, d * np.sqrt(3) / 2, 0],
                        [0, 0, 6]],
                  pbc=True)

    magmoms = [[3, 3, 0], [3, -1, 0], [-4, 0, 1.0]]

    calc = GPAW(mode=PW(400),
                symmetry='off',
                mixer=MixerDif(),
                experimental={'magmoms': magmoms},
                kpts=(4, 4, 1))
    atoms.calc = calc
    atoms.get_potential_energy()

    _, m_av = calc.dft.magmoms()
    check(m_av)

    calc.write('Cr3.gpw')
    calc = GPAW('Cr3.gpw', txt=None)

    _, m_av = calc.dft.magmoms()
    check(m_av)
