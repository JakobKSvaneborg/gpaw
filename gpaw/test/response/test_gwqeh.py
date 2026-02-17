"""Test GWQEHCorrection for van der Waals heterostructures.

Tests the G Delta-W QEH method which computes quasiparticle corrections
due to modified dielectric screening in van der Waals heterostructures.

Reference: Winther and Thygesen, 2D Materials 4, 025059 (2017).
"""

import os

import numpy as np
import pytest

from ase.build import mx2
from ase.units import Hartree

from gpaw import GPAW, PW, FermiDirac


def create_mos2_groundstate(filename, kpts=(3, 3, 1), ecut=200, nbands=20):
    """Create a MoS2 monolayer groundstate calculation."""
    structure = mx2(formula='MoS2', kind='2H', a=3.184, thickness=3.127,
                    size=(1, 1, 1), vacuum=5.0)

    calc = GPAW(mode=PW(ecut),
                parallel={'domain': 1},
                xc='LDA',
                kpts={'size': kpts, 'gamma': True},
                occupations=FermiDirac(0.01),
                txt=filename.replace('.gpw', '.txt'))

    structure.calc = calc
    structure.get_potential_energy()
    calc.diagonalize_full_hamiltonian(nbands=nbands)
    calc.write(filename, 'all')
    return filename


def create_chi_building_block(gpwfile, chi_filename, ecut=10, q_max=0.5):
    """Create a chi building block file for QEH calculations."""
    from gpaw.response.df import DielectricFunction
    from gpaw.response.qeh import QEHChiCalc

    df = DielectricFunction(calc=gpwfile,
                            frequencies={'type': 'nonlinear',
                                         'omegamax': 5,
                                         'domega0': 0.1,
                                         'omega2': 0.5},
                            ecut=ecut,
                            rate=0.01,
                            truncation='2D')

    chicalc = QEHChiCalc(df)
    chicalc.save_chi_npz(q_max=q_max, filename=chi_filename)


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


# ---------- Integration tests (require DFT + QEH) ----------


@pytest.fixture(scope='module')
def mos2_gpw_and_chi(module_tmp_path):
    """Shared fixture: MoS2 groundstate and chi building block.

    Uses module scope to avoid recomputing the expensive DFT groundstate
    and dielectric building block for each test.
    """
    pytest.importorskip('qeh')
    gpwfile = str(module_tmp_path / 'MoS2.gpw')
    chi_file = str(module_tmp_path / 'MoS2-chi.npz')

    create_mos2_groundstate(gpwfile, kpts=(3, 3, 1), ecut=200, nbands=20)
    create_chi_building_block(gpwfile, chi_file, ecut=10, q_max=0.5)

    return gpwfile, chi_file


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_monolayer_qp_correction(mos2_gpw_and_chi):
    """Test QP corrections for an isolated MoS2 monolayer.

    For an isolated monolayer (single building block, no neighbors),
    Delta W = W_HS - W_mono should be zero, and therefore the
    QP correction should vanish.
    """
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile, chi_file = mos2_gpw_and_chi

    # Isolated monolayer: structure has just one layer
    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_mono',
                          kpts=[0],
                          bands=(8, 12),
                          structure=[chi_file],
                          d=[6.15],
                          layer=0,
                          domega0=0.1,
                          omega2=5.0)

    sigma_sin, dsigma_sin = gwq.calculate_QEH()
    qp_sin = gwq.calculate_qp_correction()

    # Check shapes
    assert sigma_sin.shape == (1, 1, 4)  # (nspins, nkpts, nbands)
    assert qp_sin.shape == (1, 1, 4)

    # For a single isolated layer, Delta W = W_HS - W_mono = 0.
    # Therefore the self-energy correction and QP correction must vanish.
    np.testing.assert_allclose(sigma_sin, 0.0, atol=1e-10)
    np.testing.assert_allclose(qp_sin, 0.0, atol=1e-10)


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_bilayer_physical_signs(mos2_gpw_and_chi):
    """Test that bilayer QP corrections have physically correct signs.

    In a bilayer, the additional screening from the neighboring layer
    reduces the band gap. The image charge interaction is attractive,
    which means:
      - Valence band correction > 0 (VBM shifts up)
      - Conduction band correction < 0 (CBM shifts down)

    Reference: Winther and Thygesen, 2D Materials 4, 025059 (2017).
    """
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile, chi_file = mos2_gpw_and_chi

    # Bilayer MoS2 with typical interlayer distance
    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_bilayer',
                          kpts=[0],
                          bands=(8, 12),
                          structure=[chi_file, chi_file],
                          d=[6.15],
                          layer=0,
                          domega0=0.1,
                          omega2=5.0)

    qp_sin = gwq.calculate_qp_correction()

    # MoS2 bands 8,9 are valence, 10,11 are conduction (LDA, 18 electrons)
    # The occupations at kpt=0 tell us which are occupied:
    valence_correction = qp_sin[0, 0, :2]   # bands 8, 9
    conduction_correction = qp_sin[0, 0, 2:]  # bands 10, 11

    # Physical requirement: band gap is reduced by environmental screening.
    # Valence bands shift UP (positive correction),
    # conduction bands shift DOWN (negative correction).
    assert np.all(valence_correction > 0), \
        f'Valence corrections should be > 0, got {valence_correction}'
    assert np.all(conduction_correction < 0), \
        f'Conduction corrections should be < 0, got {conduction_correction}'

    # The total band gap reduction should be non-trivial.
    # For MoS2 bilayer at d=6.15 Ang, the gap reduction is typically
    # on the order of 0.1-0.5 eV (depending on convergence).
    gap_reduction = valence_correction.max() - conduction_correction.min()
    assert gap_reduction > 0.01, \
        f'Band gap reduction too small: {gap_reduction:.4f} eV'


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_bilayer_vs_trilayer(mos2_gpw_and_chi):
    """Test that more layers produce larger QP corrections.

    Adding more screening layers should increase |Delta W| and
    therefore increase the magnitude of the QP correction. The
    band gap reduction in a trilayer should be larger than in a bilayer.
    """
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile, chi_file = mos2_gpw_and_chi

    # Bilayer
    gwq_bi = GWQEHCorrection(calc=gpwfile,
                              filename='gwqeh_bi',
                              kpts=[0],
                              bands=(8, 12),
                              structure=[chi_file, chi_file],
                              d=[6.15],
                              layer=0,
                              domega0=0.1,
                              omega2=5.0)
    qp_bi = gwq_bi.calculate_qp_correction()

    # Trilayer (target layer sandwiched between two layers)
    gwq_tri = GWQEHCorrection(calc=gpwfile,
                               filename='gwqeh_tri',
                               kpts=[0],
                               bands=(8, 12),
                               structure=[chi_file, chi_file, chi_file],
                               d=[6.15, 6.15],
                               layer=1,
                               domega0=0.1,
                               omega2=5.0)
    qp_tri = gwq_tri.calculate_qp_correction()

    # Trilayer corrections should be larger in magnitude
    # (more screening -> larger gap reduction)
    gap_reduction_bi = (qp_bi[0, 0, :2].max()
                        - qp_bi[0, 0, 2:].min())
    gap_reduction_tri = (qp_tri[0, 0, :2].max()
                         - qp_tri[0, 0, 2:].min())
    assert gap_reduction_tri > gap_reduction_bi, \
        (f'Trilayer gap reduction ({gap_reduction_tri:.4f} eV) should exceed '
         f'bilayer ({gap_reduction_bi:.4f} eV)')


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_state_file(mos2_gpw_and_chi):
    """Test saving and loading state files for restart capability."""
    from gpaw.response.gwqeh import GWQEHCorrection

    gpwfile, chi_file = mos2_gpw_and_chi

    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_restart',
                          kpts=[0],
                          bands=(8, 10),
                          structure=[chi_file],
                          d=[6.15],
                          layer=0,
                          domega0=0.1,
                          omega2=5.0)

    qp_original = gwq.calculate_qp_correction()
    assert os.path.exists('gwqeh_restart_qeh.npz')

    # Restart from file
    gwq2 = GWQEHCorrection(calc=gpwfile,
                           filename='gwqeh_restart',
                           kpts=[0],
                           bands=(8, 10),
                           structure=[chi_file],
                           d=[6.15],
                           layer=0,
                           domega0=0.1,
                           omega2=5.0,
                           restart=True)

    assert gwq2.complete is True
    qp_reloaded = gwq2.calculate_qp_correction()
    np.testing.assert_allclose(qp_original, qp_reloaded)
