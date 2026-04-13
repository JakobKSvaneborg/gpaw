"""Test saving and loading of the precomputed screened interaction W_qGG.

The precompute_W/W_file interface should yield a BSE Hamiltonian that is
identical (up to floating-point noise) to one where W is computed on the
fly.
"""

import numpy as np
import pytest

from gpaw.response.bse import BSE, ScreenedPotential


_BSE_KWARGS = dict(
    ecut=50,
    valence_bands=2,
    conduction_bands=2,
    nbands=16,
    mode='BSE',
    truncation='2D',
)


@pytest.mark.response
def test_bse_save_load_W(in_tmp_dir, gpw_files, mpi):
    """precompute_W + W_file reproduces the fresh BSE Hamiltonian."""
    gpwfile = gpw_files['mos2_5x5_pw']
    comm = mpi.comm

    # Reference H_sS from a BSE where W is computed on the fly.
    bse_ref = BSE(gpwfile, comm=comm, **_BSE_KWARGS)
    H_ref_sS = bse_ref.get_bse_matrix(optical=True).H_sS
    H_ref_SS = bse_ref.collect_A_SS(H_ref_sS)

    # Precompute and save W_qGG to disk.
    bse_precomp = BSE(gpwfile, comm=comm, **_BSE_KWARGS)
    bse_precomp.precompute_W('W.pckl')

    # Fresh BSE loaded from the saved file.
    bse_loaded = BSE(gpwfile, comm=comm, W_file='W.pckl', **_BSE_KWARGS)
    H_loaded_sS = bse_loaded.get_bse_matrix(optical=True).H_sS
    H_loaded_SS = bse_loaded.collect_A_SS(H_loaded_sS)

    if comm.rank == 0:
        assert np.allclose(H_ref_SS, H_loaded_SS, atol=1e-12, rtol=1e-12), (
            f'max |H_ref - H_loaded| = '
            f'{np.max(np.abs(H_ref_SS - H_loaded_SS)):.2e}')


@pytest.mark.response
def test_bse_W_file_header_mismatch(in_tmp_dir, gpw_files, mpi):
    """Loading a W_file with incompatible settings raises ValueError."""
    gpwfile = gpw_files['mos2_5x5_pw']
    comm = mpi.comm

    bse_precomp = BSE(gpwfile, comm=comm, **_BSE_KWARGS)
    bse_precomp.precompute_W('W.pckl')

    # Change ecut -> header validation must reject.
    bad_kwargs = dict(_BSE_KWARGS)
    bad_kwargs['ecut'] = _BSE_KWARGS['ecut'] + 10
    bse_bad = BSE(gpwfile, comm=comm, W_file='W.pckl', **bad_kwargs)

    with pytest.raises(ValueError, match='W_file'):
        # Header is validated lazily on first W access, so we have to
        # trigger the direct kernel path.
        bse_bad.get_bse_matrix(optical=True)


@pytest.mark.response
def test_screened_potential_file_roundtrip(in_tmp_dir, mpi):
    """ScreenedPotential I/O methods round-trip plain arrays correctly."""
    comm = mpi.comm
    path = 'W.pckl'

    # Write a synthetic file with two q-points. pawcorr/qpd are opaque
    # to the file format, so we stub them with simple objects.
    metadata = {
        'ecut': 1.23,
        'nbands': 4,
        'integrate_gamma': {'type': 'reciprocal', 'reduced': False, 'N': 100},
        'q0_correction': False,
        'truncation': None,
        'ibzk_kc': np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        'nibzkpts': 2,
        'N_c': np.array([2, 1, 1]),
    }
    ScreenedPotential.write_header(path, metadata, comm=comm)

    W0 = np.arange(9, dtype=complex).reshape(3, 3)
    W1 = (1 + 1j) * np.eye(3, dtype=complex)

    ScreenedPotential.append_qpoint(
        path, iq=0, q_c=[0.0, 0.0, 0.0], W_GG=W0,
        pawcorr='paw0', qpd='qpd0', comm=comm)
    ScreenedPotential.append_qpoint(
        path, iq=1, q_c=[0.5, 0.0, 0.0], W_GG=W1,
        pawcorr='paw1', qpd='qpd1', comm=comm)

    reader = ScreenedPotential.open_for_reading(path, comm=comm)
    assert reader.header['version'] == ScreenedPotential._FILE_VERSION
    assert reader.header['ecut'] == 1.23

    # Dict-like lookup: order-independent.
    W, paw, qpd = reader.get([0.5, 0.0, 0.0])
    assert paw == 'paw1'
    assert qpd == 'qpd1'
    assert np.array_equal(W, W1)

    W, paw, qpd = reader.get([0.0, 0.0, 0.0])
    assert paw == 'paw0'
    assert qpd == 'qpd0'
    assert np.array_equal(W, W0)

    # Unknown q_c raises ValueError.
    with pytest.raises(ValueError, match='not found'):
        reader.get([0.25, 0.0, 0.0])
