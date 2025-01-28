import numpy as np

_complex_float = {
    np.float32: np.complex64,
    np.float64: np.complex128,
    np.complex64: np.complex64,
    np.complex128: np.complex128,
    float: complex,
    complex: complex
}

_real_float = {
    np.complex64: np.float32,
    np.complex128: np.float64,
    np.float32: np.float32,
    np.float64: np.float64,
    complex: float,
    float: float
}


def is_real_float(dtype):
    return np.issubdtype(dtype, np.floating)


def is_complex_float(dtype):
    return np.issubdtype(dtype, np.complexfloating)


def as_complex_float(dtype):
    return np.dtype(_complex_float[np.dtype(dtype).type])


def as_real_float(dtype):
    return np.dtype(_real_float[np.dtype(dtype).type])
