import pytest

from gpaw.test.fuzz import main


@pytest.mark.serial
def test_fuzz():
    error = main('')
    assert error == 0
