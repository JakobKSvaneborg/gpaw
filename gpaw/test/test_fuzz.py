import pytest

import gpaw.test.fuzz


def dft(atoms, **kwargs):
    raise ValueError('Nice error message')


@pytest.mark.serial
def test_fuzz(monkeypatch):
    monkeypatch.setattr(gpaw.test.fuzz, 'DFT', dft)
    error = gpaw.test.fuzz.main('--seed=13 -n1')
    assert error == 0
