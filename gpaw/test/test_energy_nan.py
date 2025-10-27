from ase import Atoms
import numpy as np
import pytest

from gpaw import GPAW, KohnShamConvergenceError
from gpaw.mpi import world
from gpaw.new.extensions import Extension


class EnergyNaNifier(Extension):
    name = 'energy_nanifier'

    def get_energy_contributions(self):
        return {'nan_energy': np.nan}


@pytest.mark.skipif(world.size > 1, reason='Gamma-point calculation.')
def test_energy_nan(gpaw_new):
    if not gpaw_new:
        pytest.skip('Only GPAW new')

    a = 4.0
    atom = Atoms('H', cell=(a, a, a), pbc=True)
    atom.center()
    atom.calc = GPAW(mode={'name': 'pw', 'ecut': 200},
                     extensions=[EnergyNaNifier()])
    with pytest.raises(KohnShamConvergenceError, match='Some energy terms*'):
        atom.get_potential_energy()
