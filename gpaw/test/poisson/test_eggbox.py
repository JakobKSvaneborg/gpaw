from ase import Atoms
from gpaw import GPAW
import numpy as np


def test_eggbox():
    atoms = Atoms('H', cell=[3, 3, 3], pbc=True)
    atoms.calc = GPAW(mode='pw',
                      poissonsolver={'fast': True},
                      convergence={'energy': 1e-6},
                      symmetry='off',
                      txt='1b')
    E = []
    X = np.linspace(0, 0.9, 30)
    # X = [0, 0.03, 0]
    for x in X:
        atoms.positions[0, 0] = x
        e = atoms.get_potential_energy()
        # print(x, e)
        E.append(e)
        if 0:  # x > 0:
            break
    if 1:
        import matplotlib.pyplot as plt
        plt.plot(X, E)
        plt.show()


test_eggbox()
