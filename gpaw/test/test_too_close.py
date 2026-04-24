"""Make sure we get an exception when an atom is too close to the boundary."""
import pytest
from ase import Atoms

from gpaw.old.grid_descriptor import GridBoundsError
from gpaw.utilities import AtomsTooClose


@pytest.mark.parametrize('mode', ['fd', 'pw'])
def test_too_close_to_boundary(mode, mpi):
    a = 4.0
    x = 0.1
    hydrogen = Atoms('H', [(x, x, x)],
                     cell=(a, a, a),
                     pbc=(1, 1, 0))
    hydrogen.calc = mpi.GPAW(mode=mode)
    with pytest.raises((GridBoundsError, AtomsTooClose)):
        hydrogen.get_potential_energy()
