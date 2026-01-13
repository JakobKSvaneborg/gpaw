"""Test GWQEHCorrection for van der Waals heterostructures."""

import numpy as np
import pytest

from ase.build import mx2
from ase.units import Hartree

from gpaw import GPAW, PW, FermiDirac
from gpaw.mpi import world


def create_mos2_groundstate(filename, kpts=(3, 3, 1), ecut=200):
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
    calc.diagonalize_full_hamiltonian(nbands=20)
    calc.write(filename, 'all')
    return filename


def create_chi_building_block(gpwfile, chi_filename, ecut=10, q_max=0.5):
    """Create a chi building block file for QEH calculations."""
    qeh = pytest.importorskip('qeh')

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


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_initialization(in_tmp_dir):
    """Test basic initialization of GWQEHCorrection."""
    pytest.importorskip('qeh')
    from gpaw.response.gwqeh import GWQEHCorrection

    # Create groundstate
    gpwfile = 'MoS2_test.gpw'
    create_mos2_groundstate(gpwfile, kpts=(2, 2, 1), ecut=150)

    # Create building block
    chi_file = 'MoS2-chi.npz'
    create_chi_building_block(gpwfile, chi_file, ecut=8, q_max=0.4)

    # Initialize GWQEHCorrection
    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_test',
                          kpts=[0],
                          bands=(8, 12),
                          structure=[chi_file],
                          d=[6.15],
                          layer=0)

    # Check basic attributes are set correctly
    assert gwq.bands == (8, 12)
    assert gwq.kpts == [0]
    assert gwq.nspins == 1
    assert gwq.nbands > 0
    assert gwq.shape == (1, 1, 4)  # (nspins, nkpts, nbands)


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_frequency_grid(in_tmp_dir):
    """Test the frequency grid generation for GWQEH."""
    from gpaw.response.gwqeh import frequency_grid

    # Test frequency grid generation
    domega0 = 0.025 / Hartree
    omega2 = 10.0 / Hartree
    omegamax = 20.0 / Hartree

    omega_w = frequency_grid(domega0, omega2, omegamax)

    # Check grid properties
    assert len(omega_w) > 0
    assert omega_w[0] == 0.0
    assert omega_w[-1] <= omegamax * 1.1  # Allow some tolerance
    # Check grid is monotonically increasing
    assert np.all(np.diff(omega_w) > 0)


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_interlayer_conversion(in_tmp_dir):
    """Test interlayer distance to thickness conversion."""
    from gpaw.response.gwqeh import interlayer_to_thickness

    # Test with uniform spacing
    d = np.array([6.0, 6.0, 6.0])
    t = interlayer_to_thickness(d)
    assert len(t) == 4
    assert t[0] == 6.0
    assert t[-1] == 6.0
    np.testing.assert_allclose(t[1:-1], 6.0)

    # Test with non-uniform spacing
    d = np.array([5.0, 7.0])
    t = interlayer_to_thickness(d)
    assert len(t) == 3
    assert t[0] == 5.0
    assert t[-1] == 7.0
    assert t[1] == pytest.approx(6.0)  # (5+7)/2


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_calculate_qeh(in_tmp_dir):
    """Test the calculate_QEH method."""
    pytest.importorskip('qeh')
    from gpaw.response.gwqeh import GWQEHCorrection

    # Create groundstate with minimal settings for speed
    gpwfile = 'MoS2_qeh.gpw'
    create_mos2_groundstate(gpwfile, kpts=(2, 2, 1), ecut=150)

    # Create building block
    chi_file = 'MoS2-chi.npz'
    create_chi_building_block(gpwfile, chi_file, ecut=8, q_max=0.4)

    # Initialize and run calculation
    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_calc',
                          kpts=[0],
                          bands=(8, 10),
                          structure=[chi_file],
                          d=[6.15],
                          layer=0,
                          domega0=0.1,
                          omega2=5.0)

    sigma_sin, dsigma_sin = gwq.calculate_QEH()

    # Check output shapes
    assert sigma_sin.shape == gwq.shape
    assert dsigma_sin.shape == gwq.shape
    assert gwq.complete is True


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_qp_correction(in_tmp_dir):
    """Test the calculate_qp_correction method."""
    pytest.importorskip('qeh')
    from gpaw.response.gwqeh import GWQEHCorrection

    # Create groundstate
    gpwfile = 'MoS2_qp.gpw'
    create_mos2_groundstate(gpwfile, kpts=(2, 2, 1), ecut=150)

    # Create building block
    chi_file = 'MoS2-chi.npz'
    create_chi_building_block(gpwfile, chi_file, ecut=8, q_max=0.4)

    # Initialize and calculate QP correction
    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_qp',
                          kpts=[0],
                          bands=(8, 10),
                          structure=[chi_file],
                          d=[6.15],
                          layer=0,
                          domega0=0.1,
                          omega2=5.0)

    qp_sin = gwq.calculate_qp_correction()

    # Check output
    assert qp_sin.shape == gwq.shape
    # QP corrections should be real (returned in eV)
    assert np.isreal(qp_sin).all() or np.allclose(qp_sin.imag, 0)

    # Placeholder reference values - to be filled in after running
    # These are example values that will need to be updated with actual results
    # expected_qp = np.array([[[0.05, 0.03]]])  # Example values in eV
    # np.testing.assert_allclose(qp_sin, expected_qp, rtol=0.1)


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_heterostructure(in_tmp_dir):
    """Test GWQEH for a simple bilayer heterostructure."""
    pytest.importorskip('qeh')
    from gpaw.response.gwqeh import GWQEHCorrection

    # Create groundstate
    gpwfile = 'MoS2_hs.gpw'
    create_mos2_groundstate(gpwfile, kpts=(2, 2, 1), ecut=150)

    # Create building block (using same for both layers in this test)
    chi_file = 'MoS2-chi.npz'
    create_chi_building_block(gpwfile, chi_file, ecut=8, q_max=0.4)

    # Test bilayer with layer=0
    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_bilayer',
                          kpts=[0],
                          bands=(8, 10),
                          structure=[chi_file, chi_file],
                          d=[6.15],
                          layer=0,
                          domega0=0.1,
                          omega2=5.0)

    qp_sin = gwq.calculate_qp_correction()
    assert qp_sin.shape == gwq.shape


@pytest.mark.response
@pytest.mark.serial
@pytest.mark.slow
def test_gwqeh_state_file(in_tmp_dir):
    """Test saving and loading state files for restart capability."""
    pytest.importorskip('qeh')
    from gpaw.response.gwqeh import GWQEHCorrection

    # Create groundstate
    gpwfile = 'MoS2_restart.gpw'
    create_mos2_groundstate(gpwfile, kpts=(2, 2, 1), ecut=150)

    # Create building block
    chi_file = 'MoS2-chi.npz'
    create_chi_building_block(gpwfile, chi_file, ecut=8, q_max=0.4)

    # Run initial calculation
    gwq = GWQEHCorrection(calc=gpwfile,
                          filename='gwqeh_restart',
                          kpts=[0],
                          bands=(8, 10),
                          structure=[chi_file],
                          d=[6.15],
                          layer=0,
                          domega0=0.1,
                          omega2=5.0)

    qp_sin_original = gwq.calculate_qp_correction()

    # Check state file was created
    import os
    assert os.path.exists('gwqeh_restart_qeh.npz')

    # Create new instance with restart
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

    # Should load from file
    assert gwq2.complete is True
    qp_sin_reloaded = gwq2.calculate_qp_correction()

    np.testing.assert_allclose(qp_sin_original, qp_sin_reloaded)
