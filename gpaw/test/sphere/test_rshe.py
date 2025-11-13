"""Test functionality to expand a function on a spherical grid in real
spherical harmonics."""

import pytest
import itertools

import numpy as np

from gpaw.atom.radialgd import EquidistantRadialGridDescriptor
from gpaw.sphere.lebedev import Y_nL
from gpaw.sphere.rshe import (RealSphericalHarmonicsExpansion,
                              calculate_reduced_rshe)


def generate_combinations():
    """Generate combinations of one and two spherical harmonic indices.
    """
    L_L = np.arange(Y_nL.shape[1])
    oneL_i = list(itertools.combinations(L_L, 1))
    twoL_i = list(itertools.combinations(L_L, 2))

    Lcomb_i = oneL_i + twoL_i

    return Lcomb_i


@pytest.mark.parametrize('Lcomb', generate_combinations())
def test_rshe(Lcomb):
    """Test the ability to regenerate a given function based on a
    real spherical harmonics expansion."""

    # Build the angular dependence of the function on the Lebedev quadrature
    f_n = np.zeros(Y_nL.shape[0])
    for L in Lcomb:
        f_n += Y_nL[:, L]
    # Build the radial dependence as a Lorentzian
    rgd = EquidistantRadialGridDescriptor(h=0.2,  # grid spacing
                                          N=11)
    f_g = 0.25 / (np.pi * (0.25**2 + rgd.r_g**2.))
    f_ng = f_n[:, np.newaxis] * f_g[np.newaxis]

    # Test the real spherical harmonics expansion without basis reduction
    rshe = RealSphericalHarmonicsExpansion.from_spherical_grid(rgd, f_ng, Y_nL)
    assert rshe.evaluate_on_quadrature() == pytest.approx(f_ng)

    # Test the ability to reduce the expansion via minimum weights
    rshe, _ = calculate_reduced_rshe(rgd, f_ng, Y_nL, wmin=1e-8)
    assert len(rshe.L_M) == len(Lcomb)
    assert all([L in rshe.L_M for L in Lcomb])
    assert all([int(np.sqrt(L)) in rshe.l_M for L in Lcomb])
    assert rshe.evaluate_on_quadrature() == pytest.approx(f_ng)

    # Test the ability to reduce the expansion by an lmax
    Lmax = max(Lcomb)
    lmax = int(np.ceil(np.sqrt(Lmax + 1) - 1))
    if lmax < 4:
        rshe, _ = calculate_reduced_rshe(rgd, f_ng, Y_nL, lmax=lmax)
        assert len(rshe.L_M) == (lmax + 1)**2
        assert rshe.evaluate_on_quadrature() == pytest.approx(f_ng)
