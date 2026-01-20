"""
Tests that our Cupy -> C++ array type conversions work as intended.
"""
import pytest
import gpaw.cgpaw as cgpaw

try:
    from gpaw.cgpaw import gpu as cgpu  # noqa: F401
except ImportError:
    pytest.skip("No _gpaw.gpu module", allow_module_level=True)

import numpy as np
from gpaw.gpu import cupy as cp, cupy_is_fake
if cupy_is_fake:
    pytest.skip("Fake cupy", allow_module_level=True)


def test_cupy_array_input():
    """Tests that we can call a C++ function that takes in
    gpaw::PyDeviceArray using Cupy array (should have automatic type cast).
    """
    arr = cp.empty(2)
    cgpaw.gpu.test_cupy_input(arr)


def test_bad_input():
    """Test that function taking in gpaw::PyDeviceArray won't work with
    non-Cupy array input.
    """
    with pytest.raises(TypeError):
        cgpaw.gpu.test_cupy_input(np.empty(2))

    with pytest.raises(TypeError):
        cgpaw.gpu.test_cupy_input(2)

    with pytest.raises(TypeError):
        cgpaw.gpu.test_cupy_input("cool")

    # fake Cupy is not supported either (should we allow it?!)
    from gpaw.gpu import cpupy
    with pytest.raises(TypeError):
        cgpaw.gpu.test_cupy_input(cpupy.zeros(2))


def test_array_metadata():
    """Test that the type conversion correctly interprets array metadata.
    """
    a = cp.empty((1))
    cgpaw.gpu.test_array_metadata(a, a)

    a = cp.empty((1, 2, 3, 4))
    cgpaw.gpu.test_array_metadata(a, a)

    a = cp.empty((2, 1, 2), order='F')
    cgpaw.gpu.test_array_metadata(a, a)
