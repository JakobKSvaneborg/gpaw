import numpy as np
import pytest
from ase import Atoms
from gpaw import GPAW
from gpaw.new.extensions import Extension


class EnergyNaNifier(Extension):
    name = 'energy_nanifier'

    def get_energy_contributions(self):
        return {'nan_energy': np.nan}


@pytest.mark.serial
def test_energy_nan():
    a = 4.0
    atom = Atoms('H', cell=(a, a, a), pbc=True)
    atom.center()
    atom.calc = GPAW(
        mode={'name': 'pw', 'ecut': 200},
        extensions=[EnergyNaNifier()])
    with pytest.raises(ValueError, match='Some energy terms*'):
        atom.get_potential_energy()
