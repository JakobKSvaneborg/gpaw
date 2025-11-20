import numpy as np
import pytest
import scipy

from gpaw.gpu import cupy as cp
from gpaw.gpu import cupy_is_fake
from gpaw.gpu import cupyx as cupyx

"""Tests for Cupy's FFT routines (in practice, cuFFT/hipFFT).
In principle it shouldn't be our responsibility to test these,
however, hipFFT in particular is known to be unreliable on ROCm < 6.2.
As of writing this, Cupy has offical (experimental) support only for ROCm 5.0.
So we manually test the FFT at least until Cupy's ROCm support matures.
"""


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason="No need to test with CPUPY")
@pytest.mark.parametrize(
    "shape",
    [
        (24, 50, 70),  # OK on rocm 6.0.2, fails on < 6(?)
        (294, 294, 108),  # OK on rocm 6.2.2, fails on 6.0.2
    ],
)
def test_cupy_rfftn(shape: tuple):
    """
    Test real-to-complex FFT and its inverse.
    """

    rng = cp.random.default_rng(42)
    arr = rng.random(shape)

    try:
        res = cupyx.scipy.fft.rfftn(arr)
        # Should give back the original array:
        res_inverse = cupyx.scipy.fft.irfftn(res, shape)

    except cp.cuda.cufft.CuFFTError as e:
        raise RuntimeError(
            "Cupy rfftn failed, known to happen for ROCm < 6.2") from e

    arr_np = cp.asnumpy(arr)
    ref = scipy.fft.rfftn(arr_np)

    res = cp.asnumpy(res)
    res_inverse = cp.asnumpy(res_inverse)

    np.testing.assert_allclose(res, ref, atol=1e-12)
    np.testing.assert_allclose(res_inverse, arr_np, atol=1e-12)


@pytest.mark.gpu
@pytest.mark.skipif(cupy_is_fake, reason="No need to test with CPUPY")
@pytest.mark.parametrize(
    "shape",
    [
        (24, 50, 70),
        (294, 294, 108),
    ],
)
def test_cupy_fftn(shape: tuple):
    """
    Test complex-to-complex FFT and its inverse. AFAIK we have no reports of
    these failing, but tested here for completeness.
    """

    rng = cp.random.default_rng(42)
    arr = rng.random(shape) + 1j * rng.random(shape)

    try:
        res = cupyx.scipy.fft.fftn(arr)
        # Should give back the original array:
        res_inverse = cupyx.scipy.fft.ifftn(res, shape)

    except cp.cuda.cufft.CuFFTError as e:
        raise RuntimeError(
            "Cupy fftn failed, please report this to GPAW developers") from e

    arr_np = cp.asnumpy(arr)
    ref = scipy.fft.fftn(arr_np)

    res = cp.asnumpy(res)
    res_inverse = cp.asnumpy(res_inverse)

    np.testing.assert_allclose(res, ref, atol=1e-12)
    np.testing.assert_allclose(res_inverse, arr_np, atol=1e-12)
