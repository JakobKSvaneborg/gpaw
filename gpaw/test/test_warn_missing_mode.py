"""Make sure we get a warning when mode is not supplied."""
import pytest
from ase.build import molecule

from gpaw import GPAW
from gpaw.old.calculator import DeprecatedParameterWarning


@pytest.mark.ci
def test_no_mode_supplied(gpaw_new: bool, mpi) -> None:
    if gpaw_new:
        with pytest.raises(TypeError):
            mpi.GPAW()
        return
    a = 6.0
    hydrogen = molecule('H2', cell=[a, a, a])
    hydrogen.center()
    with pytest.warns(DeprecatedParameterWarning):
        hydrogen.calc = mpi.GPAW()
        hydrogen.calc.initialize(hydrogen)
