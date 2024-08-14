import pytest
from ase import Atoms
from gpaw import GPAW


@pytest.mark.serial
def test_hse06():
    atoms = Atoms('Li2', [[0, 0, 0], [0, 0, 2.0]])
    atoms.center(vacuum=2.5)
    atoms.calc = GPAW(mode='pw', xc='HSE06', nbands=3)
    e = atoms.get_potential_energy()
    eigs = atoms.calc.get_eigenvalues(spin=0)
    assert e == pytest.approx(-5.633278, abs=1e-3)
    assert eigs[0] == pytest.approx(-4.67477532, abs=1e-3)


# @pytest.mark.serial
def test_h():
    atoms = Atoms('H', magmoms=[1])
    atoms.center(vacuum=2.5)
    atoms.calc = GPAW(mode='pw',
                      xc='HSE06',
                      nbands=2,
                      convergence={'energy': 1e-4})
    e = atoms.get_potential_energy()
    eigs = atoms.calc.get_eigenvalues(spin=0)
    assert e == pytest.approx(-1.703969, abs=1e-3)
    assert eigs[0] == pytest.approx(-9.71440, abs=1e-3)
