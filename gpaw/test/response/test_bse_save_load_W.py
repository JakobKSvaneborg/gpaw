"""Tests for precomputing and reloading the BSE screened interaction W.

Covers the three behaviours of the ``BSE.precompute_W(path)`` /
``BSE(..., W_file=path)`` round trip:

1. a reloaded run produces a Hamiltonian that is bit-identical to the
   on-the-fly run,
2. ``precompute_W`` is restartable: a partial pre-compute over one
   q-point, followed by a second call with no restriction, computes only
   the remaining q-points,
3. loading with incompatible cache metadata raises ``ValueError``.
"""
from pathlib import Path

import numpy as np
import pytest

from gpaw.response.bse import BSE


def _make_bse(gpwfile, comm, W_file=None):
    return BSE(gpwfile,
               q_c=(0, 0, 0),
               soc_tol=0.01,
               add_soc=True,
               ecut=10,
               valence_bands=2,
               conduction_bands=2,
               eshift=0.8,
               nbands=15,
               mode='BSE',
               truncation='2D',
               W_file=W_file,
               comm=comm)


def _count_cached_qpoints(path):
    """Number of per-q-point W entries in the cache (metadata excluded)."""
    files = list(Path(path).glob('cache.*.json'))
    keys = [p.stem.split('.', 1)[1] for p in files]
    return sum(1 for k in keys if k != 'metadata')


def _hamiltonian(gpwfile, comm, W_file=None):
    bse = _make_bse(gpwfile, comm, W_file=W_file)
    matrix = bse.get_bse_matrix(optical=True)
    # H_sS is distributed across ranks; the comparison is
    # per-rank and bit-exact.
    return np.array(matrix.H_sS, copy=True)


@pytest.mark.response
def test_bse_W_file_round_trip(in_tmp_dir, gpw_files, mpi):
    comm = mpi.comm
    gpwfile = gpw_files['mos2_5x5_pw']

    reference = _hamiltonian(gpwfile, comm)

    cache_path = Path('W_cache')
    _make_bse(gpwfile, comm).precompute_W(cache_path)

    loaded = _hamiltonian(gpwfile, comm, W_file=cache_path)

    np.testing.assert_array_equal(loaded, reference)


@pytest.mark.response
def test_bse_W_file_partial_precompute_and_restart(
        in_tmp_dir, gpw_files, mpi):
    comm = mpi.comm
    gpwfile = gpw_files['mos2_5x5_pw']

    reference = _hamiltonian(gpwfile, comm)

    cache_path = Path('W_cache')

    # First partial pass: only iq=0.
    _make_bse(gpwfile, comm).precompute_W(cache_path, qpoints=[0])
    comm.barrier()
    if comm.rank == 0:
        assert _count_cached_qpoints(cache_path) == 1
    comm.barrier()

    # Second pass: should compute only the remaining q-points.
    bse = _make_bse(gpwfile, comm)
    nibzkpts = bse.qd.nibzkpts
    assert nibzkpts > 1, (
        'This test requires more than one IBZ q-point to be meaningful; '
        f'got {nibzkpts}.')

    # Count invocations of the (expensive) chi0 builder to prove that the
    # second pass only recomputes the remaining q-points.
    call_count = [0]
    original_calculate = bse._chi0calc.calculate

    def counting_calculate(q_c):
        call_count[0] += 1
        return original_calculate(q_c)

    bse._chi0calc.calculate = counting_calculate
    bse.precompute_W(cache_path)
    comm.barrier()

    if comm.rank == 0:
        assert _count_cached_qpoints(cache_path) == nibzkpts
        assert call_count[0] == nibzkpts - 1
    comm.barrier()

    loaded = _hamiltonian(gpwfile, comm, W_file=cache_path)
    np.testing.assert_array_equal(loaded, reference)


@pytest.mark.response
def test_bse_W_file_header_mismatch_raises(in_tmp_dir, gpw_files, mpi):
    comm = mpi.comm
    gpwfile = gpw_files['mos2_5x5_pw']

    cache_path = Path('W_cache')
    _make_bse(gpwfile, comm).precompute_W(cache_path)

    bse = BSE(gpwfile,
              q_c=(0, 0, 0),
              soc_tol=0.01,
              add_soc=True,
              ecut=8,  # different ecut -> cache metadata mismatch
              valence_bands=2,
              conduction_bands=2,
              eshift=0.8,
              nbands=15,
              mode='BSE',
              truncation='2D',
              W_file=cache_path,
              comm=comm)

    with pytest.raises(ValueError, match='different BSE settings'):
        bse.get_bse_matrix(optical=True)
