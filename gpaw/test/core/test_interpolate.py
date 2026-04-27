import pytest

from gpaw.core import UGDesc


@pytest.mark.ci
def test_fft_interpolation():
    a = UGDesc(cell=[1, 1, 1], size=(4, 4, 4)).zeros()
    b = UGDesc(cell=[1, 1, 1], size=(8, 8, 8)).zeros()
    a.data[2, 2, 2] = 1.0
    a.interpolate(out=b)
    assert (b.data[::2, ::2, ::2] == a.data).all()

    b.fft_restrict(out=a)
    assert a.integrate() == pytest.approx(b.integrate(), abs=1e-12)
