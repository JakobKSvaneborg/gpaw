import numpy as np
import pytest
from ase.units import Bohr

from gpaw import GPAW
from gpaw.sphere.integrate import (default_spherical_drcut,
                                   find_volume_conserving_lambd,
                                   integrate_lebedev,
                                   periodic_truncation_function, radial_trapz,
                                   radial_truncation_function,
                                   spherical_truncation_function_collection)


def generate_analytical_integrals():
    """Return pair of functions that (1) evaluates a given f(r) (2) integrates

    r_cut
    /
    | r^2 dr f(r)
    /
    0

    analytically together with a relative tolerance.
    """

    def linear(r):
        f = 2. - r
        f[r > 2.] = 0.
        return f

    def integrate_linear(rcut):
        if rcut > 2.:
            return 4. / 3.
        else:
            return (2 / 3. - rcut / 4.) * rcut**3.

    def gaussian(r):
        return np.exp(-r**2 / 4)

    def integrate_gaussian(rcut):
        from scipy.special import erf
        return 2 * np.sqrt(np.pi) * erf(rcut / 2) \
            - 2 * rcut * np.exp(-rcut**2 / 4)

    def lorentzian(r):
        return 1 / (r**2 + 4)

    def integrate_lorentzian(rcut):
        return rcut - 2. * np.arctan(rcut / 2)

    return [
        # Exact numerical integration with linear interpolation
        (linear, integrate_linear, 1e-8),
        # Approximate numerical integration
        (gaussian, integrate_gaussian, 1e-5),
        (lorentzian, integrate_lorentzian, 1e-5)
    ]


@pytest.mark.parametrize('analytical_integral',
                         generate_analytical_integrals())
def test_radial_trapz(analytical_integral):
    func, integrate, rtol = analytical_integral
    # Create radial grid and evaluate the function
    r_g = np.linspace(0., 3.0, 250)
    f_g = func(r_g)

    # Compute the radial integral for a number of different cutoffs:
    for ncut in np.arange(100, 260, 10):
        rcut = r_g[ncut - 1]
        ref = integrate(rcut)
        result = radial_trapz(f_g[:ncut], r_g[:ncut])
        assert abs(result - ref) <= 1e-8 + rtol * ref


@pytest.mark.parametrize('rc', np.linspace(0.5, 4.5, 9))
def test_smooth_truncation_function(rc):
    # Define radial grid
    r_g = np.linspace(0., 5.0, 501)
    # Calculate spherical volume corresponding to the cutoff
    ref = 4 * np.pi * rc**3. / 3.
    # Test drc from grid spacing x2 to grid spacing x50
    for drc in np.linspace(0.02, 0.5, 10):
        theta_g = radial_truncation_function(r_g, rc, drc)
        # Calculate spherical volume with truncation function
        vol = 4 * np.pi * radial_trapz(theta_g, r_g)
        assert abs(vol - ref) <= 1e-8 + 1e-4 * ref


def test_fe_augmentation_sphere(gpw_files):
    # Extract the spherical grid information from the iron fixture
    calc = GPAW(gpw_files['fe_pw'], txt=None)
    setup = calc.setups[0]
    rgd = setup.rgd
    Y_nL = setup.xc_correction.Y_nL

    # Create a function which is unity over the entire grid
    f_ng = np.ones((Y_nL.shape[0], rgd.N))

    # Integrate f(r) with different cutoffs, to check that the volume is
    # correctly recovered
    r_g = rgd.r_g
    for rcut in np.linspace(0.1 / Bohr, np.max(r_g), 100):
        ref = 4 * np.pi * rcut**3. / 3.

        # Do standard numerical truncation
        # Integrate angular components, then radial
        f_g = integrate_lebedev(f_ng)
        vol = rgd.integrate_trapz(f_g, rcut=rcut)
        assert abs(vol - ref) <= 1e-8 + 1e-6 * ref
        # Integrate radial components, then angular
        f_n = rgd.integrate_trapz(f_ng, rcut=rcut)
        vol = integrate_lebedev(f_n)
        assert abs(vol - ref) <= 1e-8 + 1e-6 * ref

        # Integrate f(r) θ(r<rc) using a smooth truncation function
        if rcut > np.max(setup.rcut_j):
            # This method relies on a sufficiently dense grid sampling to be
            # accurate, so we only test values inside the augmentation sphere
            continue
        theta_g = radial_truncation_function(r_g, rcut)
        ft_ng = f_ng * theta_g[np.newaxis]
        # Integrate angular components, then radial
        ft_g = integrate_lebedev(ft_ng)
        vol = rgd.integrate_trapz(ft_g)
        assert abs(vol - ref) <= 1e-8 + 1e-4 * ref
        # Integrate radial components, then angular
        ft_n = rgd.integrate_trapz(ft_ng)
        vol = integrate_lebedev(ft_n)
        assert abs(vol - ref) <= 1e-8 + 1e-4 * ref


def test_fe_periodic_truncation_function(gpw_files):
    # Extract the grid information from the iron fixture
    calc = GPAW(gpw_files['fe_pw'], txt=None)
    finegd = calc.density.finegd
    spos_ac = calc.spos_ac

    # Integrate θ(r) with different cutoffs, to check that the sphere volume
    # is correctly recovered
    a = 2.867  # lattice constant in Å
    rcut_max = 2 * a / (3 * Bohr)  # 2a / 3 in Bohr
    # Get default drcut corresponding to the coarse real-space grid
    drcut = default_spherical_drcut(calc.density.gd)
    for rcut in np.linspace(rcut_max / 6, rcut_max, 13):
        ref = 4 * np.pi * rcut**3. / 3.

        # Optimize λ-parameter, generate θ(r) and integrate
        lambd = find_volume_conserving_lambd(rcut, drcut)
        theta_r = periodic_truncation_function(finegd, spos_ac[0],
                                               rcut, drcut, lambd)
        vol = finegd.integrate(theta_r)
        assert abs(vol - ref) <= 1e-8 + 1e-2 * ref

    # Make sure that the difference between coarse and fine grid drcut is not
    # too large
    finedrcut_vol = finegd.integrate(
        periodic_truncation_function(finegd, spos_ac[0], rcut))
    assert abs(vol - finedrcut_vol) / finedrcut_vol < 1e-3
    # Make sure that we get a different value numerically, if we change drcut
    diff_vol = finegd.integrate(
        periodic_truncation_function(finegd, spos_ac[0], rcut,
                                     drcut=drcut * 1.5))
    assert abs(vol - diff_vol) > 1e-8
    # Test that the actual value of the integral changes, if we use a different
    # λ-parameter
    diff_vol = finegd.integrate(
        periodic_truncation_function(finegd, spos_ac[0], rcut,
                                     lambd=0.75))
    assert abs(vol - diff_vol) > 1e-2 * ref


def test_co_spherical_truncation_function_collection(gpw_files):
    # Extract grid information from the cobalt fixture
    calc = GPAW(gpw_files['co_pw'], txt=None)
    finegd = calc.density.finegd
    spos_ac = calc.spos_ac
    drcut = default_spherical_drcut(calc.density.gd)

    # Generate collection of spherical truncation functions with varrying rcut
    a = 2.5071
    c = 4.0695
    nn_dist = min(a, np.sqrt(a**2 / 3 + c**2 / 4))
    rcut_j = np.linspace(drcut, 2 * nn_dist / 3, 13)
    rcut_aj = [rcut_j, rcut_j]
    stfc = spherical_truncation_function_collection(finegd, spos_ac, rcut_aj,
                                                    drcut=drcut)

    # Integrate collection of spherical truncation functions
    ones_r = finegd.empty()
    ones_r[:] = 1.
    vol_aj = {0: np.empty(len(rcut_j)), 1: np.empty(len(rcut_j))}
    stfc.integrate(ones_r, vol_aj)

    # Check that the integrated volume matches the spherical volume and an
    # analogous manual integration
    for a, spos_c in enumerate(spos_ac):
        for rcut, vol in zip(rcut_j, vol_aj[a]):
            ref = 4 * np.pi * rcut**3. / 3.
            assert abs(vol - ref) <= 1e-8 + 1e-2 * ref

            # "Manual" integration
            theta_r = periodic_truncation_function(finegd, spos_c, rcut, drcut)
            manual_vol = finegd.integrate(theta_r)
            assert abs(vol - manual_vol) <= 1e-8 + 1e-6 * ref
