import numpy as np
import pytest

from gpaw.mpi import serial_comm
from gpaw.response import bse as bse_module
from gpaw.response.bse import BSEBackend, BSEMatrix


class DummyContext:
    def __init__(self):
        self.comm = serial_comm
        self.timer = DummyTimer()

    def print(self, *args, **kwargs):
        pass


class DummyTimer:
    def start(self, name):
        return None

    def stop(self):
        return None


class DummyBSE:
    def __init__(self, nS):
        self.context = DummyContext()
        self.nS = nS
        self.ns = nS


@pytest.mark.response
def test_exclude_states_serial_skips_blacs(monkeypatch):
    def fail_blacs_grid(*args, **kwargs):
        raise AssertionError('BlacsGrid should not be used in serial mode')

    monkeypatch.setattr(bse_module, 'BlacsGrid', fail_blacs_grid)

    H = np.arange(36, dtype=float).reshape(6, 6)
    mat = BSEMatrix(df_S=np.ones(6), H_sS=H, deps_S=np.zeros(6), deps_max=1.0)
    bse = DummyBSE(nS=6)
    exclude = np.array([1, 4])

    H_rr, new_desc = mat.exclude_states(bse, exclude)

    expected = np.delete(np.delete(H, exclude, axis=0), exclude, axis=1)
    assert new_desc is None
    assert np.array_equal(H_rr, expected)


@pytest.mark.response
def test_get_chi_wgg_tda_serial_skips_blacs(monkeypatch):
    def fail_blacs_grid(*args, **kwargs):
        raise AssertionError('BlacsGrid should not be used in serial mode')

    monkeypatch.setattr(bse_module, 'BlacsGrid', fail_blacs_grid)

    class DummyGS:
        volume = 1.0

    class DummyBSE:
        def __init__(self):
            self.context = DummyContext()
            self.use_tammdancoff = True
            self.nS = 3
            self.eig_data = (np.array([1.0, 2.0, 3.0]),
                             np.eye(3, dtype=complex),
                             np.array([], dtype=int))
            self.rho_SG = np.array([[1.0, 0.5],
                                    [0.2, 1.0],
                                    [0.1, 0.3]], dtype=complex)
            self.df_S = np.array([1.0, 0.5, 0.25], dtype=float)
            self.gs = DummyGS()

        def _cache_eig_data(self, irreducible, optical, w_w):
            return None

    bse = DummyBSE()
    chi_wGG = BSEBackend.get_chi_wGG(bse,
                                     w_w=np.array([0.0, 1.0]),
                                     eta=0.2,
                                     optical=True,
                                     irreducible=False)
    assert chi_wGG.shape == (2, 2, 2)
