import pytest
from ase import Atoms
from ase.units import Ha

from gpaw.new.ase_interface import GPAW


@pytest.mark.serial
@pytest.mark.xfail
def test_h_exx_lcao():
    atoms = Atoms('H', magmoms=[1])
    atoms.center(vacuum=2.5)
    atoms.calc = GPAW(mode='lcao', xc='EXX')
    atoms.get_potential_energy()
    eig = atoms.calc.get_eigenvalues(spin=0)[0]
    assert eig == pytest.approx(-0.5 * Ha, abs=0.05)
