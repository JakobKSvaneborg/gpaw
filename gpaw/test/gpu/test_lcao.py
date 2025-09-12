import pytest
from ase import Atoms

from gpaw.dft import GPAW


@pytest.mark.gpu
@pytest.mark.serial
def test_h2():
    h2 = Atoms('H2',
               [[0, 0, 0],
                [0, 0, 0.75]],
               pbc=False)
    h2.center(vacuum=2.5)
    h2.calc = GPAW(
        mode='lcao',
        parallel={'gpu': True},
        experimental={'pw_pot_calc': True})
    h2.get_potential_energy()
