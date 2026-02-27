"""Test GWQEHCorrection for van der Waals heterostructures.

Tests the G Delta-W QEH method which computes quasiparticle corrections
due to modified dielectric screening in van der Waals heterostructures.

Reference: Winther and Thygesen, 2D Materials 4, 025059 (2017).
"""

import os

import numpy as np
import pytest

from ase.units import Hartree


# ---------- Unit tests (fast, no DFT needed) ----------


def test_gwqeh_frequency_grid():
    """Test the non-linear frequency grid generation.

    The grid should start at zero, be monotonically increasing, and
    become denser near omega=0 (spacing ~ domega0) and sparser at
    high frequencies.
    """
    from gpaw.response.gwqeh import frequency_grid

    domega0 = 0.025 / Hartree
    omega2 = 10.0 / Hartree
    omegamax = 20.0 / Hartree

    omega_w = frequency_grid(domega0, omega2, omegamax)

    # Grid starts at zero
    assert omega_w[0] == 0.0
    # Grid is monotonically increasing
    dw = np.diff(omega_w)
    assert np.all(dw > 0)
    # First spacing should be close to domega0
    assert dw[0] == pytest.approx(domega0, rel=0.01)
    # Spacing increases with frequency (non-linear grid)
    assert dw[-1] > dw[0]
    # Grid covers the requested range
    assert omega_w[-1] >= omegamax * 0.9


def test_gwqeh_interlayer_conversion():
    """Test interlayer distance to layer thickness conversion.

    For N layers with N-1 interlayer distances d[i], the layer
    thicknesses t[i] are: t[0]=d[0], t[-1]=d[-1], and
    t[i] = (d[i-1] + d[i]) / 2 for interior layers.
    """
    from gpaw.response.gwqeh import interlayer_to_thickness

    # Uniform spacing: all thicknesses should equal the spacing
    d = np.array([6.0, 6.0, 6.0])
    t = interlayer_to_thickness(d)
    assert len(t) == 4
    np.testing.assert_allclose(t, 6.0)

    # Non-uniform spacing: interior layers are averages
    d = np.array([5.0, 7.0])
    t = interlayer_to_thickness(d)
    assert len(t) == 3
    assert t[0] == pytest.approx(5.0)
    assert t[1] == pytest.approx(6.0)  # (5+7)/2
    assert t[2] == pytest.approx(7.0)

    # Single interlayer distance (bilayer)
    d = np.array([6.15])
    t = interlayer_to_thickness(d)
    assert len(t) == 2
    np.testing.assert_allclose(t, 6.15)

    # Invalid input
    with pytest.raises(ValueError):
        interlayer_to_thickness(np.array([]))


# ---------- Helpers for synthetic Delta-W ----------


def _make_synthetic_dW(nq=50, nw=100, amplitude=0.005):
    """Create synthetic Delta-W(q,omega) arrays for testing without qeh.

    Models the change in screened interaction due to a neighboring
    layer: Delta-W = W_HS - W_mono < 0 (more screening reduces W).
    Uses Lorentzian frequency dependence and exponential q-decay.

    Returns qqeh (Bohr^-1), wqeh (Hartree), dW_qw (complex).
    """
    qqeh = np.linspace(0, 3.0, nq)   # Bohr^-1
    wqeh = np.linspace(0, 3.0, nw)   # Hartree

    q = qqeh[:, np.newaxis]
    w = wqeh[np.newaxis, :]

    # Delta-W < 0: additional screening reduces the screened interaction
    dW = -amplitude * np.exp(-q) / (1 + w**2)
    return qqeh, wqeh, dW.astype(complex)


# ---------- Integration tests (DFT + synthetic Delta-W, no qeh) ----------


@pytest.mark.response
@pytest.mark.serial
def test_gwqeh_zero_dW(in_tmp_dir, gpw_files):
    """Delta-W = 0 must give zero QP correction.

    When there is no change in screening (isolated monolayer),
    the self-energy correction must vanish identically.
    """
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])
    qqeh, wqeh, dW_qw = _make_synthetic_dW()
    dW_zero = np.zeros_like(dW_qw)

    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_zero',
                          kpts=[0],
                          bands=(8, 12),
                          dW_qw=dW_zero,
                          qqeh=qqeh,
                          wqeh=wqeh,
                          domega0=0.1,
                          omega2=5.0)

    sigma_sin, dsigma_sin = gwq.calculate_QEH()
    qp_sin = gwq.calculate_qp_correction()

    # Check shapes: (nspins, nkpts, nbands) = (1, 1, 4)
    assert sigma_sin.shape == (1, 1, 4)
    assert qp_sin.shape == (1, 1, 4)

    # Zero Delta-W must give zero correction
    np.testing.assert_allclose(sigma_sin, 0.0, atol=1e-10)
    np.testing.assert_allclose(qp_sin, 0.0, atol=1e-10)


@pytest.mark.response
@pytest.mark.serial
def test_gwqeh_linearity_in_dW(in_tmp_dir, gpw_files):
    """Self-energy is linear in Delta-W.

    The GW self-energy Sigma = i G W is linear in W. Therefore
    doubling Delta-W must exactly double the QP correction.
    """
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])
    qqeh, wqeh, dW_qw = _make_synthetic_dW()

    gwq1 = GWQEHCorrection(calc=gpwfile,
                           filename='gwqeh_1x',
                           kpts=[0],
                           bands=(8, 12),
                           dW_qw=dW_qw,
                           qqeh=qqeh,
                           wqeh=wqeh,
                           domega0=0.1,
                           omega2=5.0)
    qp1 = gwq1.calculate_qp_correction()

    gwq2 = GWQEHCorrection(calc=gpwfile,
                           filename='gwqeh_2x',
                           kpts=[0],
                           bands=(8, 12),
                           dW_qw=2.0 * dW_qw,
                           qqeh=qqeh,
                           wqeh=wqeh,
                           domega0=0.1,
                           omega2=5.0)
    qp2 = gwq2.calculate_qp_correction()

    # Corrections should be non-zero to make this test meaningful
    assert np.any(np.abs(qp1) > 1e-12), \
        'Corrections are zero; linearity test is vacuous'

    # Exact linearity (self-energy is linear in W)
    np.testing.assert_allclose(qp2, 2.0 * qp1, rtol=1e-10)


@pytest.mark.response
@pytest.mark.serial
def test_gwqeh_bilayer_physical_signs(in_tmp_dir, gpw_files):
    """Bilayer QP corrections have physically correct signs.

    With Delta-W < 0 (additional screening from neighbor), the band gap
    is reduced: valence bands shift up (positive correction) and
    conduction bands shift down (negative correction).

    Reference: Winther and Thygesen, 2D Materials 4, 025059 (2017).
    """
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])
    qqeh, wqeh, dW_qw = _make_synthetic_dW()

    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_signs',
                          kpts=[0],
                          bands=(8, 12),
                          dW_qw=dW_qw,
                          qqeh=qqeh,
                          wqeh=wqeh,
                          domega0=0.1,
                          omega2=5.0)

    qp_sin = gwq.calculate_qp_correction()

    # MoS2: 18 valence electrons -> 9 occupied bands
    # bands (8,12): bands 8,9 are valence; bands 10,11 are conduction
    valence_correction = qp_sin[0, 0, :2]
    conduction_correction = qp_sin[0, 0, 2:]

    # Physical signs: Delta-W < 0 -> gap reduction
    assert np.all(valence_correction > 0), \
        f'Valence corrections should be > 0, got {valence_correction}'
    assert np.all(conduction_correction < 0), \
        f'Conduction corrections should be < 0, got {conduction_correction}'


@pytest.mark.response
@pytest.mark.serial
def test_gwqeh_state_file(in_tmp_dir, gpw_files):
    """State files enable correct restart."""
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])
    qqeh, wqeh, dW_qw = _make_synthetic_dW()

    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_restart',
                          kpts=[0],
                          bands=(8, 10),
                          dW_qw=dW_qw,
                          qqeh=qqeh,
                          wqeh=wqeh,
                          domega0=0.1,
                          omega2=5.0)

    qp_original = gwq.calculate_qp_correction()
    assert os.path.exists('gwqeh_restart_qeh.npz')

    # Restart from state file
    gwq2 = GWQEHCorrection(calc=gpwfile,
                           filename='gwqeh_restart',
                           kpts=[0],
                           bands=(8, 10),
                           dW_qw=dW_qw,
                           qqeh=qqeh,
                           wqeh=wqeh,
                           domega0=0.1,
                           omega2=5.0,
                           restart=True)

    assert gwq2.complete is True
    qp_reloaded = gwq2.calculate_qp_correction()
    np.testing.assert_allclose(qp_original, qp_reloaded)


# ---------- Full pipeline test (requires qeh package) ----------


@pytest.fixture(scope='module')
def mos2_chi(module_tmp_path, gpw_files):
    """Create chi building block for full-pipeline tests.

    Requires the external ``qeh`` package.  The chi building block
    is computed once per test module.
    """
    pytest.importorskip('qeh')
    from gpaw.response.df import DielectricFunction
    from gpaw.response.qeh import QEHChiCalc

    gpwfile = gpw_files['mos2_pw_fulldiag']
    chi_file = str(module_tmp_path / 'MoS2-chi.npz')

    df = DielectricFunction(calc=gpwfile,
                            frequencies={'type': 'nonlinear',
                                         'omegamax': 5,
                                         'domega0': 0.1,
                                         'omega2': 0.5},
                            ecut=10,
                            rate=0.01,
                            truncation='2D')
    chicalc = QEHChiCalc(df)
    chicalc.save_chi_npz(q_max=0.5, filename=chi_file)

    return str(gpwfile), chi_file


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_full_pipeline(in_tmp_dir, mos2_chi):
    """Full pipeline: DFT -> building block -> QEH -> QP correction.

    Tests the complete workflow including the QEH building block
    computation.  Verifies that a bilayer produces physically
    correct signs for the QP correction.
    """
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile, chi_file = mos2_chi

    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_full',
                          kpts=[0],
                          bands=(8, 12),
                          structure=[chi_file, chi_file],
                          d=[6.15],
                          layer=0,
                          domega0=0.1,
                          omega2=5.0)

    qp_sin = gwq.calculate_qp_correction()

    # Physical signs for bilayer screening
    valence_correction = qp_sin[0, 0, :2]
    conduction_correction = qp_sin[0, 0, 2:]
    assert np.all(valence_correction > 0), \
        f'Valence corrections should be > 0, got {valence_correction}'
    assert np.all(conduction_correction < 0), \
        f'Conduction corrections should be < 0, got {conduction_correction}'
