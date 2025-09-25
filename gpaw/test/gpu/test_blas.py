import numpy as np
import pytest

from gpaw.gpu import cupy as cp, cupy_is_fake
from gpaw.utilities.blas import (gpu_axpy, gpu_dotc, gpu_dotu, gpu_gemm,
                                 gpu_gemv, gpu_mmm, gpu_r2k, gpu_rk, gpu_scal,
                                 mmm, r2k, rk)


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason='No cupy')
@pytest.mark.parametrize('dtype', [float, complex])
def test_blas(dtype, set_device):
    from cupy.cuda.stream import Stream
    stream = Stream(null=True)
    stream.use()
    N = 100
    rng = np.random.default_rng(seed=42)
    a = np.zeros((N, N), dtype=dtype)
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    x = np.zeros((N,), dtype=dtype)
    y = np.zeros_like(x)
    if dtype == float:
        a[:] = rng.random((N, N))
        b[:] = rng.random((N, N))
        c[:] = rng.random((N, N))
        x[:] = rng.random((N,))
        y[:] = rng.random((N,))
    else:
        a.real = rng.random((N, N))
        a.imag = rng.random((N, N))
        b.real = rng.random((N, N))
        b.imag = rng.random((N, N))
        c.real = rng.random((N, N))
        c.imag = rng.random((N, N))
        x.real = rng.random((N,))
        x.imag = rng.random((N,))
        y.real = rng.random((N,))
        y.imag = rng.random((N,))

    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    c_gpu = cp.asarray(c)
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)

    def approx(y):
        return pytest.approx(y, rel=1e-14, abs=1e-14)

    # axpy
    y += 0.5 * x
    gpu_axpy(0.5, x_gpu, y_gpu)
    assert approx(y) == y_gpu.get()

    # mmm
    mmm(0.5, a, 'N', b, 'N', 0.2, c)
    gpu_mmm(0.5, a_gpu, 'N', b_gpu, 'N', 0.2, c_gpu)
    assert approx(c_gpu.get()) == c

    # gemm
    c *= 0.2
    c += 0.5 * b @ a
    gpu_gemm(0.5, a_gpu, b_gpu, 0.2, c_gpu)
    assert approx(a_gpu.get()) == a

    # gemv
    y *= 0.2
    y += 0.5 * a @ x
    gpu_gemv(0.5, a_gpu, x_gpu, 0.2, y_gpu)
    assert approx(y_gpu.get()) == y

    # rk
    rk(0.5, a, 0.2, c)
    gpu_rk(0.5, a_gpu, 0.2, c_gpu)
    assert approx(c_gpu.get()) == c

    # r2k
    from cupy.cuda.stream import Stream
    stream = Stream(non_blocking=True)
    with stream:
        c_gpu_ref = c_gpu.copy()
        c_ref = c.copy()
        r2k(0.5, a, b, 0.2, c)
        gpu_r2k(0.5, a_gpu, b_gpu, 0.2, c_gpu)
        assert approx(c_gpu.get()) == c

        # r2k sliced
        bs = 11
        for i in range(0, (N + bs - 1) // bs):
            gpu_r2k(0.5, a_gpu[:, i * bs:(i + 1) * bs],
                    b_gpu[::, i * bs:(i + 1) * bs],
                    0.2 if (i == 0) else 1.0, c_gpu_ref)
            r2k(0.5, a[:, i * bs:(i + 1) * bs],
                b[:, i * bs:(i + 1) * bs],
                0.2 if (i == 0) else 1.0, c_ref)

        assert approx(c_gpu_ref.get()) == c
        assert approx(c_ref) == c

    # dotc
    check_cpu = x.conj() @ y
    check_gpu = gpu_dotc(x_gpu, y_gpu)
    assert check_cpu == approx(check_gpu)

    # dotu
    check_cpu = x @ y
    check_gpu = gpu_dotu(x_gpu, y_gpu)
    assert check_cpu == approx(check_gpu)

    # scal
    a *= 0.5
    gpu_scal(0.5, a_gpu)
    assert approx(a_gpu.get()) == a
