import pytest
import numpy as np

from gpaw.mpi import world
from gpaw.nlopt.basic import NLOData


def generate_testing_data(ns, nk, nb, seed=42):
    rng = np.random.default_rng(seed=42)

    w_sk = rng.random((ns, nk))
    f_skn = rng.random((ns, nk, nb))
    E_skn = rng.random((ns, nk, nb))
    p_skvnn = rng.random((ns, nk, 3, nb, nb)) \
        + 1j * rng.random((ns, nk, 3, nb, nb))

    return w_sk, f_skn, E_skn, p_skvnn


@pytest.mark.skipif(world.size > 1, reason='Serial only')
def test_write_load_serial(in_tmp_dir):
    w_sk, f_skn, E_skn, p_skvnn = generate_testing_data(2, 4, 20)

    nlo = NLOData(w_sk, f_skn, E_skn, p_skvnn, world)
    nlo.write('nlodata.npz')

    newdata = NLOData.load('nlodata.npz', world)
    assert newdata.w_sk == pytest.approx(w_sk, abs=1e-16)
    assert newdata.f_skn == pytest.approx(f_skn, abs=1e-16)
    assert newdata.E_skn == pytest.approx(E_skn, abs=1e-16)
    assert newdata.p_skvnn == pytest.approx(p_skvnn, abs=1e-16)


def test_serial_file_parallel_data(in_tmp_dir):
    # Random data only on rank = 0
    if world.rank == 0:
        w_sk, f_skn, E_skn, p_skvnn = generate_testing_data(2, 4, 20)
    else:
        w_sk = None
        f_skn = None
        E_skn = None
        p_skvnn = None

    nlo = NLOData(w_sk, f_skn, E_skn, p_skvnn, world)
    nlo.write('nlodata.npz')
    k_info = nlo.distribute()

    newdata = NLOData.load('nlodata.npz', world)
    k_info_new = newdata.distribute()
    for newdata, data in zip(k_info_new.values(), k_info.values()):
        assert newdata[0] == pytest.approx(data[0], abs=1e-16)
        assert newdata[1] == pytest.approx(data[1], abs=1e-16)
        assert newdata[2] == pytest.approx(data[2], abs=1e-16)
        assert newdata[3] == pytest.approx(data[3], abs=1e-16)


def test_write_load_parallel(in_tmp_dir):
    # Same random data array on each core
    w_sk, f_skn, E_skn, p_skvnn = generate_testing_data(2, 4, 20)

    nlo = NLOData(w_sk, f_skn, E_skn, p_skvnn, world)
    nlo.write('nlodata.npz')

    newdata = NLOData.load('nlodata.npz', world)
    k_info = newdata.distribute()

    # Compare the distributed data with original data
    for u, data in k_info.items():
        s = 0 if u < w_sk.shape[1] else 1
        k = u % w_sk.shape[1]
        assert data[0] == pytest.approx(w_sk[s, k], abs=1e-16)
        assert data[1] == pytest.approx(f_skn[s, k], abs=1e-16)
        assert data[2] == pytest.approx(E_skn[s, k], abs=1e-16)
        assert data[3] == pytest.approx(p_skvnn[s, k], abs=1e-16)
