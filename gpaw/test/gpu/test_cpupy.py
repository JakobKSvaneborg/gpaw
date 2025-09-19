import numpy as np
import pytest
from gpaw.gpu import cupy as cp


@pytest.mark.gpu
@pytest.mark.serial
def test_basics():
    a = cp.empty(2)
    b = a.get()
    assert isinstance(b, np.ndarray)
    assert b.dtype == float


@pytest.mark.gpu
@pytest.mark.serial
@pytest.mark.xfail
def test_grr():
    a = cp.empty((2, 2))
    a[:] = 2.5
    b = np.float64(3.0) * a
    assert isinstance(b, cp.ndarray)


@pytest.mark.gpu
@pytest.mark.serial
def test_no_implicit_conversion():
    a = cp.ones(2)
    b = np.empty(2)
    with pytest.raises(TypeError):
        b[:] = a
