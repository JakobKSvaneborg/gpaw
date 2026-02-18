# BLAS utils for "noblas = True" builds

import numpy as np
from typing import TypeVar

T = TypeVar('T', float, complex)


def op(o, m):
    if o.upper() == 'N':
        return m
    if o.upper() == 'T':
        return m.T
    if o.upper() == 'C':
        return m.conj().T
    raise ValueError(f'unknown op: {o}')


def rk(alpha, a: np.ndarray, beta, c: np.ndarray, trans='c'):  # noqa
    """NOTE: Fills in the entire c matrix, unlike the C-blas version that only
    fills in the lower triangle. Does NOT remove imaginary parts on the final
    diagonal (unlike many BLAS libraries)."""
    if c.size == 0:
        return
    if beta == 0:
        c[:] = 0.0
    else:
        c *= beta
    if trans == 'n':
        c += alpha * a.conj().T.dot(a)
    else:
        a = a.reshape((len(a), -1))
        c += alpha * a.dot(a.conj().T)


def r2k(alpha, a, b, beta, c, trans='c'):  # noqa
    """NOTE: Fills in the entire c matrix, unlike the C-blas version that only
    fills in the lower triangle. Does NOT remove imaginary parts on the final
    diagonal (unlike many BLAS libraries)."""
    if c.size == 0:
        return
    if beta == 0.0:
        c[:] = 0.0
    else:
        c *= beta
    if trans == 'c':
        c += (alpha * a.reshape((len(a), -1))
                .dot(b.reshape((len(b), -1)).conj().T) +
                alpha * b.reshape((len(b), -1))
                .dot(a.reshape((len(a), -1)).conj().T))
    else:
        c += alpha * (a.conj().T @ b + b.conj().T @ a)


def mmm(alpha: T, a: np.ndarray, opa: str,  # noqa
        b: np.ndarray, opb: str,
        beta: T, c: np.ndarray) -> None:
    if beta == 0.0:
        c[:] = 0.0
    else:
        c *= beta
    c += alpha * op(opa, a).dot(op(opb, b))
