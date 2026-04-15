"""Tests for Wannier helpers under k-point symmetry reduction.

The GPAW side of the Wannier pipeline unfolds IBZ wave functions to
the full BZ on demand (gpaw/new/wannier.py).  These tests check that
the singular values of the overlap matrices agree - up to SCF noise -
between runs with full BZ sampling (``symmetry='off'``) and runs that
use k-point symmetry, including the non-symmorphic case.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpaw.mpi import world

pytestmark = pytest.mark.skipif(world.size > 1,
                                reason='world.size > 1')


@pytest.fixture(scope='module')
def si_calcs(tmp_path_factory):
    """Three Si calculators: no symmetry, symmorphic, full space group."""
    pytest.importorskip('gpaw.new', reason='new gpaw stack required')

    from ase.build import bulk
    from gpaw import GPAW, PW

    calcs = {}
    for name, kw in [('nosym', {'symmetry': 'off'}),
                     ('sym', {}),
                     ('nonsymm', {'symmetry': {'symmorphic': False}})]:
        atoms = bulk('Si')
        atoms.calc = GPAW(mode=PW(200), kpts=(2, 2, 2), txt=None,
                          gpts=(16, 16, 16), **kw)
        atoms.get_potential_energy()
        calcs[name] = atoms.calc
    return calcs


def _require_new_stack(calc):
    if not type(calc).__module__.startswith('gpaw.new'):
        pytest.skip('requires new gpaw calculator (GPAW_NEW=1)')


@pytest.mark.wannier
def test_singular_values_match(si_calcs):
    """Z_nn singular values match across symmetry settings.

    Full Z elements differ by per-k gauge phases; singular values do
    not, so they are the quantity to compare.
    """
    _require_new_stack(si_calcs['nosym'])

    nbands = 4
    bz = si_calcs['nosym'].get_bz_k_points()
    assert np.allclose(bz, si_calcs['sym'].get_bz_k_points())
    assert np.allclose(bz, si_calcs['nonsymm'].get_bz_k_points())

    for dirG in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        dirG = np.asarray(dirG)
        for K in range(len(bz)):
            for K1 in range(len(bz)):
                # Only examine nearest-neighbor BZ pairs along dirG.
                delta_c = bz[K1] - bz[K] - 0.5 * dirG
                if not np.allclose(delta_c, np.round(delta_c), atol=1e-3):
                    continue
                G_I = np.round(delta_c).astype(int)
                args = dict(nbands=nbands, dirG=dirG, kpoint=K,
                            nextkpoint=K1, G_I=G_I, spin=0)
                Z_nosym = si_calcs['nosym'].get_wannier_localization_matrix(
                    **args)
                Z_sym = si_calcs['sym'].get_wannier_localization_matrix(
                    **args)
                Z_nonsymm = si_calcs['nonsymm']\
                    .get_wannier_localization_matrix(**args)
                sv_nosym = np.sort(np.linalg.svd(Z_nosym, compute_uv=False))
                sv_sym = np.sort(np.linalg.svd(Z_sym, compute_uv=False))
                sv_nonsymm = np.sort(np.linalg.svd(
                    Z_nonsymm, compute_uv=False))
                # SCF convergence in these small runs is ~5e-3.
                assert np.allclose(sv_sym, sv_nosym, atol=1e-2), (
                    dirG, K, K1, sv_sym - sv_nosym)
                assert np.allclose(sv_nonsymm, sv_nosym, atol=1e-2), (
                    dirG, K, K1, sv_nonsymm - sv_nosym)


@pytest.mark.wannier
def test_unfolded_density_matches(si_calcs):
    """Sum_n |psi_nK|^2 is gauge invariant and should match across settings."""
    _require_new_stack(si_calcs['nosym'])

    nbands = 4
    bz = si_calcs['nosym'].get_bz_k_points()
    for K in range(len(bz)):
        rho = {}
        for name in ('nosym', 'sym', 'nonsymm'):
            rho_K = np.zeros((16, 16, 16))
            for n in range(nbands):
                psit = si_calcs[name].get_pseudo_wave_function(
                    band=n, kpt=K, spin=0, bz=True)
                rho_K += np.abs(psit)**2
            rho[name] = rho_K
        scale = rho['nosym'].max()
        assert np.abs(rho['sym'] - rho['nosym']).max() < 0.02 * scale
        assert np.abs(rho['nonsymm'] - rho['nosym']).max() < 0.02 * scale


@pytest.mark.wannier
def test_get_projections_match(si_calcs):
    """Gaussian-orbital projections agree (up to gauge) across settings.

    ``get_projections`` returns ``<psi_Kn | f_i>``; the squared row-norms
    ``sum_i |f_Kni|**2`` are gauge invariant per (K, n) and should match.
    """
    _require_new_stack(si_calcs['nosym'])

    from gpaw.new.wannier import get_projections

    relpos_ac = si_calcs['nosym'].atoms.get_scaled_positions()
    locfun = [(pos, 0, 1.0) for pos in relpos_ac]

    refs = {}
    for name, calc in si_calcs.items():
        f_KnB = get_projections(calc.dft.ibzwfs, locfun, spin=0)
        refs[name] = f_KnB

    # Norms are gauge invariant per (K, band).
    for name in ('sym', 'nonsymm'):
        norm_ref = (np.abs(refs['nosym']) ** 2).sum(axis=-1)
        norm_new = (np.abs(refs[name]) ** 2).sum(axis=-1)
        # Summing over bands at fixed K removes band-mixing ambiguities
        # of degenerate subspaces.
        assert np.allclose(norm_new.sum(axis=-1),
                           norm_ref.sum(axis=-1), atol=3e-2), name
