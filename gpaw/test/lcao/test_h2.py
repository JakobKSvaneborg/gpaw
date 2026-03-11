from gpaw.dft import DFT
from ase.build import molecule


def test_h2():
    atoms = molecule('H2', cell=[3, 3, 3])
    atoms.center()
    dft = DFT(atoms,
              mode='pw',
              experimental={'new_basis': True})
    dft.converge()


if __name__ == '__main__':
    test_h2()
