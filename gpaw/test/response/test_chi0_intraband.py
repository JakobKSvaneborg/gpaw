import numpy as np
import pytest

from gpaw import GPAW, PW
from gpaw.test import equal, findpeak
from gpaw.response.df import DielectricFunction
from ase.build import bulk
from ase.units import Bohr, Hartree


@pytest.mark.response
@pytest.mark.slow
def test_chi0_intraband(in_tmp_dir):
    """Comparing the plasmon peaks found in bulk sodium for two different
    atomic structures. Testing for idential plasmon peaks. Not using
    physical sodium cell."""
    a1 = bulk('Na')
    a2 = bulk('Na')
    a2.set_initial_magnetic_moments([[0.1, ]])
    a1.calc = GPAW(mode=PW(300),
                   kpts={'size': (8, 8, 8), 'gamma': True},
                   parallel={'band': 1},
                   txt='na_spinpaired.txt')
    a2.calc = GPAW(mode=PW(300),
                   kpts={'size': (8, 8, 8), 'gamma': True},
                   parallel={'band': 1},
                   txt='na_spinpol.txt')
    a1.get_potential_energy()
    a2.get_potential_energy()

    # Use twice as many bands for expanded structure
    a1.calc.diagonalize_full_hamiltonian(nbands=20)
    a2.calc.diagonalize_full_hamiltonian(nbands=20)

    a1.calc.write('intraband_spinpaired.gpw', 'all')
    a2.calc.write('intraband_spinpolarized.gpw', 'all')

    df1 = DielectricFunction('intraband_spinpaired.gpw',
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.03},
                             ecut=10,
                             rate=0.1,
                             integrationmode='tetrahedron integration',
                             txt='intraband_spinpaired_df.txt')

    df1NLFCx, df1LFCx = df1.get_dielectric_function(direction='x')
    df1NLFCy, df1LFCy = df1.get_dielectric_function(direction='y')
    df1NLFCz, df1LFCz = df1.get_dielectric_function(direction='z')
    chi0_drude = df1.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
        df1.wd, 0.1)
    wp1 = chi0_drude.plasmafreq_vv[0, 0]**0.5

    df2 = DielectricFunction('intraband_spinpaired.gpw',
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.03},
                             ecut=10,
                             rate=0.1,
                             integrationmode=None,
                             txt='intraband_spinpaired_df_im.txt')
    df2NLFCx, df2LFCx = df2.get_dielectric_function(direction='x')
    df2NLFCy, df2LFCy = df2.get_dielectric_function(direction='y')
    df2NLFCz, df2LFCz = df2.get_dielectric_function(direction='z')
    chi0_drude = df2.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
        df2.wd, 0.1)
    wp2 = chi0_drude.plasmafreq_vv[0, 0]**0.5

    df3 = DielectricFunction('intraband_spinpolarized.gpw',
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.03},
                             ecut=10,
                             rate=0.1,
                             integrationmode='tetrahedron integration',
                             txt='intraband_spinpolarized_df.txt')

    df3NLFCx, df3LFCx = df3.get_dielectric_function(direction='x')
    df3NLFCy, df3LFCy = df3.get_dielectric_function(direction='y')
    df3NLFCz, df3LFCz = df3.get_dielectric_function(direction='z')
    chi0_drude = df3.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
        df3.wd, 0.1)
    wp3 = chi0_drude.plasmafreq_vv[0, 0]**0.5

    df4 = DielectricFunction('intraband_spinpolarized.gpw',
                             frequencies={'type': 'nonlinear',
                                          'domega0': 0.03},
                             ecut=10,
                             rate=0.1,
                             integrationmode=None,
                             txt='intraband_spinpolarized_df_im.txt')

    df4NLFCx, df4LFCx = df4.get_dielectric_function(direction='x')
    df4NLFCy, df4LFCy = df4.get_dielectric_function(direction='y')
    df4NLFCz, df4LFCz = df4.get_dielectric_function(direction='z')
    chi0_drude = df4.chi0calc.chi0_opt_ext_calc.drude_calc.calculate(
        df4.wd, 0.1)
    wp4 = chi0_drude.plasmafreq_vv[0, 0]**0.5

    # Compare plasmon frequencies and intensities
    w_w = df1.wd.omega_w
    # frequency grids must be the same
    w_w2 = df2.wd.omega_w
    w_w3 = df3.wd.omega_w
    w_w4 = df4.wd.omega_w
    assert np.allclose(w_w, w_w2, atol=1e-5, rtol=1e-4)
    assert np.allclose(w_w2, w_w3, atol=1e-5, rtol=1e-4)
    assert np.allclose(w_w3, w_w4, atol=1e-5, rtol=1e-4)

    # Analytical Drude result
    n = 1 / (a1.get_volume() * Bohr**-3)

    wp = np.sqrt(4 * np.pi * n)

    # From https://doi.org/10.1021/jp810808h
    wpref = 5.71 / Hartree

    equal(wp1, wp3, 1e-2)  # spin paired matches spin polar - tetra
    equal(wp2, wp4, 1e-2)  # spin paired matches spin polar - none
    equal(wp1, wp, 0.5)  # Use larger margin when comparing to Drude
    equal(wp2, wp, 0.5)  # Use larger margin when comparing to Drude
    equal(wp1, wpref, 0.1)  # paired tetra match paper
    equal(wp2, wpref, 0.1)  # paired none match paper

    # w_x equal for paired & polarized tetra
    w1, I1 = findpeak(w_w, -(1. / df1LFCx).imag)
    w3, I3 = findpeak(w_w, -(1. / df3LFCx).imag)
    equal(w1, w3, 1e-2)
    equal(I1, I3, 1e-1)

    # w_x equal for paired & polarized none
    w2, I2 = findpeak(w_w, -(1. / df2LFCx).imag)
    w4, I4 = findpeak(w_w, -(1. / df4LFCx).imag)
    equal(w2, w4, 1e-2)
    equal(I2, I4, 1e-1)

    # w_y equal for paired & polarized tetra
    w1, I1 = findpeak(w_w, -(1. / df1LFCy).imag)
    w3, I3 = findpeak(w_w, -(1. / df3LFCy).imag)
    equal(w1, w3, 1e-2)
    equal(I1, I3, 1e-1)

    # w_y equal for paired & polarized none
    w2, I2 = findpeak(w_w, -(1. / df2LFCy).imag)
    w4, I4 = findpeak(w_w, -(1. / df4LFCy).imag)
    equal(w2, w4, 1e-2)
    equal(I2, I4, 1e-1)

    # w_z equal for paired & polarized tetra
    w1, I1 = findpeak(w_w, -(1. / df1LFCz).imag)
    w3, I3 = findpeak(w_w, -(1. / df3LFCz).imag)
    equal(w1, w3, 1e-2)
    equal(I1, I3, 1e-1)

    # w_z equal for paired & polarized none
    w2, I2 = findpeak(w_w, -(1. / df2LFCz).imag)
    w4, I4 = findpeak(w_w, -(1. / df4LFCz).imag)
    equal(w2, w4, 1e-2)
    equal(I2, I4, 1e-1)
