# fails with On entry to ZGEMV parameter number 8 had an illegal value

import pytest
from ase.build import molecule

from gpaw import GPAW, PW
from gpaw.mpi import world

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


@pytest.mark.legacy
def test_pw_moleculecg():
    m = molecule('H')
    m.center(vacuum=2.0)
    m.calc = GPAW(mode=PW(), eigensolver='cg')
    m.get_potential_energy()
