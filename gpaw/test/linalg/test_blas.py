import numpy as np
import pytest
from gpaw.utilities.blas import no_c_blas, axpy, gemmdot, mmm, mmmx, r2k, rk
from gpaw.utilities.tools import tri2full
from gpaw.utilities.blas_purepython import (
    rk as rk_purepython,
    r2k as r2k_purepython,
    mmm as mmm_purepython
)


def test_gemm_size_zero():
    c = np.ones((3, 3))
    a = np.zeros((0, 3))
    b = np.zeros((3, 0))
    d = np.zeros((0, 0))
    e = np.zeros((0, 3))
    # gemm(1.0, a, b, 0.0, c, 'n')
    mmm(1.0, b, 'N', a, 'N', 0.0, c)
    assert (c == 0.0).all()
    mmm(1.0, d, 'N', a, 'N', 0.0, e)


def test_linalg_blas():
    a = np.arange(5 * 7).reshape(5, 7) + 4.
    a2 = np.arange(3 * 7).reshape(3, 7) + 3.
    b = np.arange(7) - 2.

    # Check gemmdot with floats
    assert np.all(np.dot(a, b) == gemmdot(a, b))
    assert np.all(np.dot(a, a2.T) == gemmdot(a, a2, trans='t'))
    assert np.all(np.dot(a, a2.T) == gemmdot(a, a2, trans='c'))
    assert np.dot(b, b) == gemmdot(b, b)

    # Check gemmdot with complex arrays
    a = a * (2 + 1.j)
    a2 = a2 * (-1 + 3.j)
    b = b * (3 - 2.j)
    assert np.all(np.dot(a, b) == gemmdot(a, b))
    assert np.all(np.dot(a, a2.T) == gemmdot(a, a2, trans='t'))
    assert np.all(np.dot(a, a2.T.conj()) == gemmdot(a, a2, trans='c'))
    assert np.dot(b, b) == gemmdot(b, b, trans='n')
    assert np.dot(b, b.conj()) == gemmdot(b, b, trans='c')

    # Check gemm for transa='n'
    a2 = np.arange(7 * 5 * 1 * 3).reshape(7, 5, 1, 3) * (-1. + 4.j) + 3.
    c = np.tensordot(a, a2, [1, 0])
    mmmx(1., a, 'N', a2, 'N', -1., c)
    assert not c.any()

    # Check gemm for transa='c'
    a = np.arange(4 * 5 * 1 * 3).reshape(4, 5, 1, 3) * (3. - 2.j) + 4.
    c = np.tensordot(a, a2.conj(), [[1, 2, 3], [1, 2, 3]])
    mmmx(1., a, 'N', a2, 'C', -1., c)
    assert not c.any()

    # Check axpy
    c = 5.j * a
    axpy(-5.j, a, c)
    assert not c.any()

    # Check rk
    c = np.tensordot(a, a.conj(), [[1, 2, 3], [1, 2, 3]])
    rk(1., a, -1., c)
    tri2full(c)
    assert not c.any()

    # Check gemmdot for transa='c'
    c = np.tensordot(a, a2.conj(), [-1, -1])
    gemmdot(a, a2, beta=-1., out=c, trans='c')
    assert not c.any()

    # Check gemmdot for transa='n'
    a2.shape = 3, 7, 5, 1
    c = np.tensordot(a, a2, [-1, 0])
    gemmdot(a, a2, beta=-1., out=c, trans='n')
    assert not c.any()

    # Check r2k
    a2 = 5. * a
    c = np.tensordot(a, a2.conj(), [[1, 2, 3], [1, 2, 3]])
    r2k(.5, a, a2, -1., c)
    tri2full(c)
    assert not c.any()


@pytest.mark.skipif(no_c_blas, reason="No C-blas to compare against")
@pytest.mark.parametrize('dtype', [float, complex])
def test_purepython_blas(dtype):
    """Tests stuff from blas_purepython.py and compares the results against
    similar funcs from proper C-blas."""
    # Mostly copied from gpu/test_blas.
    N = 100
    rng = np.random.default_rng(seed=42)
    a = np.zeros((N, N), dtype=dtype)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    if dtype == float:
        a[:] = rng.random((N, N))
        b[:] = rng.random((N, N))
        c[:] = rng.random((N, N))
    else:
        a.real = rng.random((N, N))
        a.imag = rng.random((N, N))
        b.real = rng.random((N, N))
        b.imag = rng.random((N, N))
        c.real = rng.random((N, N))
        c.imag = rng.random((N, N))

    a_ref = a.copy()
    b_ref = b.copy()
    c_ref = c.copy()

    def approx(y):
        return pytest.approx(y, rel=1e-14, abs=1e-14)

    # mmm
    mmm(0.5, a_ref, 'N', b_ref, 'N', 0.2, c_ref)
    mmm_purepython(0.5, a, 'N', b, 'N', 0.2, c)
    assert approx(c_ref) == c

    """For rk and r2k we have to compare only the lower triangle of the output
    matrix 'c', because the C-blas functions only fill in this part. Meanwhile
    the Purepython versions don't make any promises like this and might just
    fill in the entire matrix."""
    # rk
    rk(0.5, a_ref, 0.2, c_ref)
    rk_purepython(0.5, a, 0.2, c)
    assert approx(np.tril(c_ref)) == np.tril(c)

    # r2k
    c_ref_bu = c_ref.copy()
    c_bu = c.copy()
    r2k(0.5, a_ref, b_ref, 0.2, c_ref)
    r2k_purepython(0.5, a, b, 0.2, c)
    assert approx(np.tril(c_ref)) == np.tril(c)

    # r2k sliced
    bs = 11
    for i in range(0, (N + bs - 1) // bs):
        r2k(0.5, a_ref[:, i * bs:(i + 1) * bs],
            b_ref[::, i * bs:(i + 1) * bs],
            0.2 if (i == 0) else 1.0, c_ref_bu)
        r2k(0.5, a[:, i * bs:(i + 1) * bs],
            b[:, i * bs:(i + 1) * bs],
            0.2 if (i == 0) else 1.0, c_bu)

    assert approx(np.tril(c_ref_bu)) == np.tril(c)
    assert approx(np.tril(c_bu)) == np.tril(c)
