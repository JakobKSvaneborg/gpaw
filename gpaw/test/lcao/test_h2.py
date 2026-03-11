from gpaw.dft import DFT
from ase.build import molecule


def test_h2(new=True):
    atoms = molecule('H2', cell=[3, 3, 3])
    atoms.center()
    atoms.pbc = True
    dft = DFT(atoms,
              mode='lcao',
              parallel={'gpu': False},
              experimental={'new_basis': new})
    dft.converge()


if __name__ == '__main__':
    import sys
    test_h2(int(sys.argv[1]))
