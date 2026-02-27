"""Test GWQEHCorrection and GWmQEHCorrection for van der Waals heterostructures.

Tests the G Delta-W QEH method which computes quasiparticle corrections
due to modified dielectric screening in van der Waals heterostructures,
including the generalized mQEH basis method.

References:
    Winther and Thygesen, 2D Materials 4, 025059 (2017).
"""

import os

import numpy as np
import pytest

from ase.units import Hartree, Bohr


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


# ---------- Helpers for mQEH synthetic data ----------


def _make_synthetic_mqeh_data(nq=50, nw=100, nbasis=1, amplitude=0.005):
    """Create synthetic mQEH Delta-W matrix and basis functions for testing.

    Generates a full set of mQEH data (Delta-W matrix, density and potential
    basis functions on a z-grid) without requiring the qeh package.

    For nbasis=1 (monopole), the density basis is a Gaussian localized at z=0
    and the potential basis is the constant function phi=1.  The Delta-W matrix
    is a scalar (1x1) matching the scalar _make_synthetic_dW.

    For nbasis=2, a dipole basis function is added.

    Returns qqeh, wqeh, dW_qw (scalar monopole), dW_qw_matrix, phi_qiz,
    drho_qzi, z_z, dz.
    """
    qqeh = np.linspace(0, 3.0, nq)   # Bohr^-1
    wqeh = np.linspace(0, 3.0, nw)   # Hartree

    q = qqeh[:, np.newaxis]
    w = wqeh[np.newaxis, :]

    # z-grid (Bohr): centered around z=0, extends +/- 15 Bohr
    nz = 200
    z_z = np.linspace(-15.0, 15.0, nz)
    dz = z_z[1] - z_z[0]

    # Density basis functions: Gaussians centered at z=0
    sigma = 1.5  # width in Bohr
    drho_monopole = np.exp(-z_z**2 / (2 * sigma**2))
    drho_monopole /= np.sqrt(np.sum(drho_monopole**2) * dz)  # normalize

    # Potential basis: constant (dual to monopole density when properly
    # normalized). For the monopole, phi=1/integral(drho) so that
    # <drho|phi> = 1 (biorthonormality).
    integral_drho = np.sum(drho_monopole) * dz
    phi_monopole = np.ones(nz) / integral_drho

    # Build arrays
    # drho_qzi: shape (nq, nz, nbasis) - density functions on z-grid
    drho_qzi = np.zeros((nq, nz, nbasis), dtype=complex)
    drho_qzi[:, :, 0] = drho_monopole[np.newaxis, :]

    # phi_qiz: shape (nq, nbasis, nz) - potential basis functions
    phi_qiz = np.zeros((nq, nbasis, nz), dtype=complex)
    phi_qiz[:, 0, :] = phi_monopole[np.newaxis, :]

    if nbasis >= 2:
        # Dipole: z * Gaussian
        drho_dipole = z_z * np.exp(-z_z**2 / (2 * sigma**2))
        drho_dipole /= np.sqrt(np.sum(drho_dipole**2) * dz)

        integral_z_drho = np.sum(drho_dipole * z_z) * dz
        phi_dipole = z_z / integral_z_drho

        drho_qzi[:, :, 1] = drho_dipole[np.newaxis, :]
        phi_qiz[:, 1, :] = phi_dipole[np.newaxis, :]

    # Delta-W matrix: shape (nq, nw, nbasis, nbasis)
    # Monopole-monopole matches the scalar dW
    dW_scalar = -amplitude * np.exp(-q) / (1 + w**2)  # (nq, nw)

    dW_qw_matrix = np.zeros((nq, nw, nbasis, nbasis), dtype=complex)
    dW_qw_matrix[:, :, 0, 0] = dW_scalar

    if nbasis >= 2:
        # Off-diagonal coupling (small)
        dW_qw_matrix[:, :, 0, 1] = 0.1 * dW_scalar
        dW_qw_matrix[:, :, 1, 0] = 0.1 * dW_scalar
        dW_qw_matrix[:, :, 1, 1] = 0.3 * dW_scalar

    return (qqeh, wqeh, dW_scalar.astype(complex),
            dW_qw_matrix, phi_qiz, drho_qzi, z_z, dz)


# ---------- mQEH integration tests (DFT + synthetic mQEH data) ----------


@pytest.mark.response
@pytest.mark.serial
def test_mqeh_zero_dW(in_tmp_dir, gpw_files):
    """mQEH: Delta-W = 0 must give zero QP correction.

    Same principle as the monopole test: no change in screening means
    no self-energy correction.
    """
    from gpaw.response.gwqeh import GWmQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])
    (qqeh, wqeh, dW_scalar,
     dW_matrix, phi_qiz, drho_qzi, z_z, dz) = _make_synthetic_mqeh_data()

    dW_zero = np.zeros_like(dW_scalar)
    dW_matrix_zero = np.zeros_like(dW_matrix)

    gwq = GWmQEHCorrection(calc=gpwfile,
                            filename='mqeh_zero',
                            kpts=[0],
                            bands=(8, 12),
                            dW_qw=dW_zero,
                            qqeh=qqeh,
                            wqeh=wqeh,
                            domega0=0.1,
                            omega2=5.0,
                            ecut_mqeh=50.0,
                            dW_qw_matrix=dW_matrix_zero,
                            phi_qiz=phi_qiz,
                            drho_qzi=drho_qzi,
                            z_z_qeh=z_z,
                            dz_qeh=dz)

    sigma_sin, dsigma_sin = gwq.calculate_QEH()
    qp_sin = gwq.calculate_qp_correction()

    assert sigma_sin.shape == (1, 1, 4)
    assert qp_sin.shape == (1, 1, 4)

    np.testing.assert_allclose(sigma_sin, 0.0, atol=1e-10)
    np.testing.assert_allclose(qp_sin, 0.0, atol=1e-10)


@pytest.mark.response
@pytest.mark.serial
def test_mqeh_linearity_in_dW(in_tmp_dir, gpw_files):
    """mQEH: Self-energy is linear in Delta-W.

    Doubling the Delta-W matrix must exactly double the correction.
    """
    from gpaw.response.gwqeh import GWmQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])
    (qqeh, wqeh, dW_scalar,
     dW_matrix, phi_qiz, drho_qzi, z_z, dz) = _make_synthetic_mqeh_data()

    gwq1 = GWmQEHCorrection(calc=gpwfile,
                             filename='mqeh_1x',
                             kpts=[0],
                             bands=(8, 12),
                             dW_qw=dW_scalar,
                             qqeh=qqeh,
                             wqeh=wqeh,
                             domega0=0.1,
                             omega2=5.0,
                             ecut_mqeh=50.0,
                             dW_qw_matrix=dW_matrix,
                             phi_qiz=phi_qiz,
                             drho_qzi=drho_qzi,
                             z_z_qeh=z_z,
                             dz_qeh=dz)
    qp1 = gwq1.calculate_qp_correction()

    gwq2 = GWmQEHCorrection(calc=gpwfile,
                             filename='mqeh_2x',
                             kpts=[0],
                             bands=(8, 12),
                             dW_qw=2.0 * dW_scalar,
                             qqeh=qqeh,
                             wqeh=wqeh,
                             domega0=0.1,
                             omega2=5.0,
                             ecut_mqeh=50.0,
                             dW_qw_matrix=2.0 * dW_matrix,
                             phi_qiz=phi_qiz,
                             drho_qzi=drho_qzi,
                             z_z_qeh=z_z,
                             dz_qeh=dz)
    qp2 = gwq2.calculate_qp_correction()

    assert np.any(np.abs(qp1) > 1e-12), \
        'Corrections are zero; linearity test is vacuous'

    np.testing.assert_allclose(qp2, 2.0 * qp1, rtol=1e-10)


@pytest.mark.response
@pytest.mark.serial
def test_mqeh_physical_signs(in_tmp_dir, gpw_files):
    """mQEH: QP corrections have physically correct signs.

    With Delta-W < 0, the band gap should be reduced: valence bands
    shift up and conduction bands shift down.
    """
    from gpaw.response.gwqeh import GWmQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])
    (qqeh, wqeh, dW_scalar,
     dW_matrix, phi_qiz, drho_qzi, z_z, dz) = _make_synthetic_mqeh_data()

    gwq = GWmQEHCorrection(calc=gpwfile,
                            filename='mqeh_signs',
                            kpts=[0],
                            bands=(8, 12),
                            dW_qw=dW_scalar,
                            qqeh=qqeh,
                            wqeh=wqeh,
                            domega0=0.1,
                            omega2=5.0,
                            ecut_mqeh=50.0,
                            dW_qw_matrix=dW_matrix,
                            phi_qiz=phi_qiz,
                            drho_qzi=drho_qzi,
                            z_z_qeh=z_z,
                            dz_qeh=dz)

    qp_sin = gwq.calculate_qp_correction()

    valence_correction = qp_sin[0, 0, :2]
    conduction_correction = qp_sin[0, 0, 2:]

    assert np.all(valence_correction > 0), \
        f'Valence corrections should be > 0, got {valence_correction}'
    assert np.all(conduction_correction < 0), \
        f'Conduction corrections should be < 0, got {conduction_correction}'


@pytest.mark.response
@pytest.mark.serial
def test_mqeh_nonzero_with_higher_basis(in_tmp_dir, gpw_files):
    """mQEH with nbasis=2 gives non-zero corrections from off-diagonal terms.

    When higher basis functions (dipole) are included in the Delta-W matrix,
    the result should differ from the monopole-only case.
    """
    from gpaw.response.gwqeh import GWmQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])

    # Monopole only (nbasis=1)
    (qqeh, wqeh, dW_scalar1,
     dW_matrix1, phi1, drho1, z_z, dz) = _make_synthetic_mqeh_data(nbasis=1)

    gwq1 = GWmQEHCorrection(calc=gpwfile,
                             filename='mqeh_nb1',
                             kpts=[0],
                             bands=(8, 12),
                             dW_qw=dW_scalar1,
                             qqeh=qqeh,
                             wqeh=wqeh,
                             domega0=0.1,
                             omega2=5.0,
                             ecut_mqeh=50.0,
                             dW_qw_matrix=dW_matrix1,
                             phi_qiz=phi1,
                             drho_qzi=drho1,
                             z_z_qeh=z_z,
                             dz_qeh=dz)
    qp1 = gwq1.calculate_qp_correction()

    # Monopole + dipole (nbasis=2)
    (qqeh, wqeh, dW_scalar2,
     dW_matrix2, phi2, drho2, z_z, dz) = _make_synthetic_mqeh_data(nbasis=2)

    gwq2 = GWmQEHCorrection(calc=gpwfile,
                             filename='mqeh_nb2',
                             kpts=[0],
                             bands=(8, 12),
                             dW_qw=dW_scalar2,
                             qqeh=qqeh,
                             wqeh=wqeh,
                             domega0=0.1,
                             omega2=5.0,
                             ecut_mqeh=50.0,
                             dW_qw_matrix=dW_matrix2,
                             phi_qiz=phi2,
                             drho_qzi=drho2,
                             z_z_qeh=z_z,
                             dz_qeh=dz)
    qp2 = gwq2.calculate_qp_correction()

    # Both should be non-zero
    assert np.any(np.abs(qp1) > 1e-12), \
        'nbasis=1 corrections are zero'
    assert np.any(np.abs(qp2) > 1e-12), \
        'nbasis=2 corrections are zero'

    # Results should differ due to off-diagonal coupling and dipole projection
    assert not np.allclose(qp1, qp2, atol=1e-12), \
        'nbasis=1 and nbasis=2 give identical results; dipole has no effect'


@pytest.mark.response
@pytest.mark.serial
def test_mqeh_ecut_convergence(in_tmp_dir, gpw_files):
    """mQEH: result changes with ecut_mqeh (more G_parallel vectors).

    A higher ecut_mqeh includes more G_parallel shells.  With a purely
    q-dependent Delta-W, additional G_parallel contributions should
    change the self-energy.
    """
    from gpaw.response.gwqeh import GWmQEHCorrection

    gpwfile = str(gpw_files['mos2_pw_fulldiag'])
    (qqeh, wqeh, dW_scalar,
     dW_matrix, phi_qiz, drho_qzi, z_z, dz) = _make_synthetic_mqeh_data()

    # Very low ecut: only G_par=0
    gwq_low = GWmQEHCorrection(calc=gpwfile,
                                filename='mqeh_ecut_low',
                                kpts=[0],
                                bands=(8, 12),
                                dW_qw=dW_scalar,
                                qqeh=qqeh,
                                wqeh=wqeh,
                                domega0=0.1,
                                omega2=5.0,
                                ecut_mqeh=0.1,
                                dW_qw_matrix=dW_matrix,
                                phi_qiz=phi_qiz,
                                drho_qzi=drho_qzi,
                                z_z_qeh=z_z,
                                dz_qeh=dz)
    qp_low = gwq_low.calculate_qp_correction()

    # Higher ecut: more G_par shells
    gwq_high = GWmQEHCorrection(calc=gpwfile,
                                 filename='mqeh_ecut_high',
                                 kpts=[0],
                                 bands=(8, 12),
                                 dW_qw=dW_scalar,
                                 qqeh=qqeh,
                                 wqeh=wqeh,
                                 domega0=0.1,
                                 omega2=5.0,
                                 ecut_mqeh=50.0,
                                 dW_qw_matrix=dW_matrix,
                                 phi_qiz=phi_qiz,
                                 drho_qzi=drho_qzi,
                                 z_z_qeh=z_z,
                                 dz_qeh=dz)
    qp_high = gwq_high.calculate_qp_correction()

    # Both should be non-zero
    assert np.any(np.abs(qp_low) > 1e-12), \
        'Low ecut corrections are zero'
    assert np.any(np.abs(qp_high) > 1e-12), \
        'High ecut corrections are zero'

    # Results should differ when more G_par vectors are included
    assert not np.allclose(qp_low, qp_high, atol=1e-12), \
        'Low and high ecut give identical results'


# ---------- Full mQEH pipeline test (requires qeh package) ----------


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_mqeh_full_pipeline(in_tmp_dir, mos2_chi):
    """Full mQEH pipeline: DFT -> building block -> mQEH -> QP correction.

    Tests the complete mQEH workflow including the QEH building block
    computation.  Verifies that a bilayer produces physically correct
    signs for the QP correction and that the result differs from the
    monopole-only calculation.
    """
    from gpaw.response.gwqeh import GWQEHCorrection, GWmQEHCorrection

    gpwfile, chi_file = mos2_chi

    # Monopole-only (original method)
    gwq_mono = GWQEHCorrection(calc=gpwfile,
                                filename='mqeh_full_mono',
                                kpts=[0],
                                bands=(8, 12),
                                structure=[chi_file, chi_file],
                                d=[6.15],
                                layer=0,
                                domega0=0.1,
                                omega2=5.0)
    qp_mono = gwq_mono.calculate_qp_correction()

    # Full mQEH method
    gwq_mqeh = GWmQEHCorrection(calc=gpwfile,
                                 filename='mqeh_full',
                                 kpts=[0],
                                 bands=(8, 12),
                                 structure=[chi_file, chi_file],
                                 d=[6.15],
                                 layer=0,
                                 domega0=0.1,
                                 omega2=5.0,
                                 ecut_mqeh=50.0)
    qp_mqeh = gwq_mqeh.calculate_qp_correction()

    # Physical signs for bilayer screening
    valence_correction = qp_mqeh[0, 0, :2]
    conduction_correction = qp_mqeh[0, 0, 2:]
    assert np.all(valence_correction > 0), \
        f'Valence corrections should be > 0, got {valence_correction}'
    assert np.all(conduction_correction < 0), \
        f'Conduction corrections should be < 0, got {conduction_correction}'

    # mQEH should differ from monopole due to G_par>0 contributions
    assert not np.allclose(qp_mono, qp_mqeh, atol=1e-12), \
        'mQEH gives identical result to monopole; G_par contributions missing'
