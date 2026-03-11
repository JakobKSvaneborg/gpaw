from gpaw.dft import DFT
from ase.build import molecule


def test_h2(new=False):
    atoms = molecule('H2', cell=[3, 3, 3])
    atoms.center()
    atoms.pbc = True
    dft = DFT(atoms,
              mode='pw',
              experimental={'new_basis': new})
    dft.converge()


if __name__ == '__main__':
    import sys
    test_h2(int(sys.argv[1]))
