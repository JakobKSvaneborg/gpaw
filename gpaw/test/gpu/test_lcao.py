import pytest
from ase import Atoms

from gpaw.dft import GPAW
from gpaw import GPAW_NO_C_EXTENSION


@pytest.mark.gpu
@pytest.mark.serial
def test_h2():
    if GPAW_NO_C_EXTENSION:
        pytest.skip('GPAW_NO_C_EXTENSION')

    h2 = Atoms('H2',
               [[0, 0, 0],
                [0, 0, 0.75]],
               pbc=False)
    h2.center(vacuum=2.5)
    h2.calc = GPAW(
        mode='lcao',
        basis='dzp',
        parallel={'gpu': True},
        experimental={'pw_pot_calc': True})
    e = h2.get_potential_energy()
    assert e == pytest.approx(-6.514335)
