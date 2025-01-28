import numpy as np
from gpaw.typing import DTypeLike

_complex_float = {
    np.float32: np.complex64,
    np.float64: np.complex128,
    np.complex64: np.complex64,
    np.complex128: np.complex128,
    float: complex,
    complex: complex}

_real_float = {
    np.complex64: np.float32,
    np.complex128: np.float64,
    np.float32: np.float32,
    np.float64: np.float64,
    complex: float,
    float: float}


def is_real_dtype(dtype: DTypeLike) -> bool:
    """Check if dtype is real.

    >>> [is_real_dtype(dt) for dt in
    ...  [float, np.float32, np.complex128]]
    [True, True, False]
    """
    return np.issubdtype(dtype, np.floating)


def is_complex_dtype(dtype):
    """Check if dtype is complex.

    >>> [is_complex_dtype(dt) for dt in
    ...  [complex, np.complex64, np.float32]]
    [True, True, False]
    """
    return np.issubdtype(dtype, np.complexfloating)


def to_complex_dtype(dtype):
    """Convert dtype to complex dtype.

    >>> [to_complex_dtype(dt) for dt in
    ...  [np.float32, np.float64, complex]]
    [dtype('complex64'), dtype('complex128'), dtype('complex128')]
    """
    return np.dtype(_complex_float[np.dtype(dtype).type])


def to_real_dtype(dtype):
    """Convert dtype to complex dtype.

    >>> [to_real_dtype(dt) for dt in
    ...  [np.float32, np.float64, complex]]
    [dtype('float32'), dtype('float64'), dtype('float64')]
    """
    return np.dtype(_real_float[np.dtype(dtype).type])
