"""Make sure we get a warning when mode is not supplied."""
import pytest


@pytest.mark.ci
def test_no_mode_supplied(mpi) -> None:
    with pytest.raises(TypeError):
        mpi.GPAW()
