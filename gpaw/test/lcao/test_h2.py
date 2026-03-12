import pytest
from ase.build import molecule
from gpaw.dft import DFT


@pytest.mark.serial
def test_h2(new=True):
    atoms = molecule('H2', cell=[3, 3, 3])
    atoms.center()
    atoms.pbc = True
    dft = DFT(atoms,
              mode='lcao',
              experimental={'new_basis': new})
    dft.converge()
    e = dft.calculate_energy()
    assert e == pytest.approx(-6.172722, abs=1e-6)


if __name__ == '__main__':
    import sys
    test_h2(int(sys.argv[1]))
