from gpaw.dft import Parameters
import pytest


def test_params():
    with pytest.raises():
        Parameters()
