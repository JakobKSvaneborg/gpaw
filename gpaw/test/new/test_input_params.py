import pytest

from gpaw.dft import Parameters


def test_params():
    with pytest.raises(TypeError):
        Parameters()
