"""Tests for Wannier helpers under k-point symmetry reduction.

The GPAW side of the Wannier pipeline unfolds IBZ wave functions to
the full BZ on demand (gpaw/new/wannier.py).  These tests check that

1. Gauge-invariant quantities (singular values, densities, projection
   norms) agree between runs with full BZ sampling and runs that use
   k-point symmetry — including the non-symmorphic case.
2. The translation-phase formula is correct element-wise (gauge-
   covariant), pinning the sign convention against future refactors.
3. The ``bz=True`` path on ``get_pseudo_wave_function`` is
   bitwise-identical to ``bz=False`` at IBZ representatives.
4. PAW projections from the lazy-caching path match an independent
   integration.
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

        if name == 'nonsymm':
            ibz = atoms.calc.dft.ibzwfs.ibz
            assert ibz.symmetries.translation_sc.any(), \
                'nonsymm fixture has no non-symmorphic ops'
            assert any(ibz.symmetries.translation_sc[s].any()
                       for s in ibz.s_K), \
                'No non-symmorphic op is used in the IBZ->BZ mapping'

        calcs[name] = atoms.calc
    return calcs


def _require_new_stack(calc):
    if not type(calc).__module__.startswith('gpaw.new'):
        pytest.skip('requires new gpaw calculator (GPAW_NEW=1)')


# ---- Gauge-invariant aggregate tests ----

@pytest.mark.wannier
def test_singular_values_match(si_calcs):
    """Z_nn singular values match across symmetry settings."""
    _require_new_stack(si_calcs['nosym'])

    nbands = 4
    bz = si_calcs['nosym'].get_bz_k_points()
    assert np.allclose(bz, si_calcs['sym'].get_bz_k_points())
    assert np.allclose(bz, si_calcs['nonsymm'].get_bz_k_points())

    for dirG in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        dirG = np.asarray(dirG)
        for K in range(len(bz)):
            for K1 in range(len(bz)):
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
def test_get_projections_gaussian_match(si_calcs):
    """Gaussian-orbital projections agree (up to gauge) across settings."""
    _require_new_stack(si_calcs['nosym'])

    from gpaw.new.wannier import get_projections

    relpos_ac = si_calcs['nosym'].atoms.get_scaled_positions()
    locfun = [(pos, 0, 1.0) for pos in relpos_ac]

    refs = {}
    for name, calc in si_calcs.items():
        f_KnB = get_projections(calc.dft.ibzwfs, locfun, spin=0)
        refs[name] = f_KnB

    for name in ('sym', 'nonsymm'):
        norm_ref = (np.abs(refs['nosym']) ** 2).sum(axis=-1)
        norm_new = (np.abs(refs[name]) ** 2).sum(axis=-1)
        assert np.allclose(norm_new.sum(axis=-1),
                           norm_ref.sum(axis=-1), atol=3e-2), name


@pytest.mark.wannier
def test_get_projections_projectors_match(si_calcs):
    """Bound-state projector branch agrees across symmetry settings."""
    _require_new_stack(si_calcs['nosym'])

    from gpaw.new.wannier import get_projections

    refs = {}
    for name, calc in si_calcs.items():
        f_Kni = get_projections(calc.dft.ibzwfs, 'projectors', spin=0)
        refs[name] = f_Kni

    for name in ('sym', 'nonsymm'):
        norm_ref = (np.abs(refs['nosym']) ** 2).sum(axis=-1)
        norm_new = (np.abs(refs[name]) ** 2).sum(axis=-1)
        assert np.allclose(norm_new.sum(axis=-1),
                           norm_ref.sum(axis=-1), atol=3e-2), name


# ---- Gauge-covariant (element-wise) tests ----

@pytest.mark.wannier
def test_translation_phase_matches_analytical_formula(si_calcs):
    """The non-symmorphic translation phase matches the analytical formula.

    For each BZ k-point reached via a non-symmorphic op, compare the
    PW coefficients from _bz_wfs (which applies the phase) to those
    from a bare PWArray.transform (which does not).  The ratio must
    equal exp(-2 pi i (k' + G) . s) with s = U^{-T} tau.
    """
    _require_new_stack(si_calcs['nonsymm'])

    from gpaw.new.wannier import _bz_wfs

    calc = si_calcs['nonsymm']
    ibzwfs = calc.dft.ibzwfs
    ibz = ibzwfs.ibz

    tested = 0
    for K in range(len(ibz.bz)):
        s = int(ibz.s_K[K])
        tau_c = ibz.symmetries.translation_sc[s]
        if not tau_c.any():
            continue
        U_cc = np.asarray(ibz.symmetries.rotation_scc[s])
        tr = bool(ibz.time_reversal_K[K])
        k_ibz = int(ibz.bz2ibz_K[K])

        wfs_ibz = ibzwfs._get_wfs(k_ibz, 0)
        psit_rot = wfs_ibz.psit_nX.transform(U_cc, complex_conjugate=tr)
        psit_bz = _bz_wfs(ibzwfs, K, 0).psit_nX

        pw = psit_bz.desc
        s_c = np.linalg.solve(U_cc.T.astype(float), tau_c)
        expected_phase_G = np.exp(
            -2j * np.pi
            * ((pw.kpt_c[:, None] + pw.indices_cG).T @ s_c))

        ratio = psit_bz.data / psit_rot.data
        for band in range(psit_bz.data.shape[0]):
            assert np.allclose(ratio[band], expected_phase_G, atol=1e-12), \
                f'K={K}, band={band}: max deviation ' \
                f'{np.abs(ratio[band] - expected_phase_G).max()}'
        tested += 1
    assert tested > 0, 'No non-symmorphic BZ k-points were tested'


@pytest.mark.wannier
def test_bz_wfs_real_space_roundtrip(si_calcs):
    """BZ periodic part matches a real-space rotation + translation.

    Compute the BZ periodic part u_BZ(r) two independent ways:
    Path A: _bz_wfs -> ifft(periodic=True).
    Path B: IBZ ifft(periodic=True) -> remap grid points via
            U^T r - tau -> conjugate if TR -> global phase correction.

    The two should agree to machine precision.
    """
    _require_new_stack(si_calcs['nonsymm'])

    from gpaw.new.wannier import _bz_wfs

    calc = si_calcs['nonsymm']
    ibzwfs = calc.dft.ibzwfs
    ibz = ibzwfs.ibz
    grid = calc.dft.density.nt_sR.desc

    tested = 0
    for K in range(len(ibz.bz)):
        s_idx = int(ibz.s_K[K])
        tau_c = ibz.symmetries.translation_sc[s_idx]
        if not tau_c.any():
            continue
        U_cc = np.asarray(ibz.symmetries.rotation_scc[s_idx])
        tr = bool(ibz.time_reversal_K[K])
        k_ibz = int(ibz.bz2ibz_K[K])

        # Path A: _bz_wfs -> periodic part
        wfs_bz = _bz_wfs(ibzwfs, K, 0)
        grid_bz = grid.new(kpt=wfs_bz.psit_nX.desc.kpt_c,
                           dtype=wfs_bz.psit_nX.desc.dtype)
        u_BZ = wfs_bz.psit_nX.ifft(grid=grid_bz, periodic=True)

        # Path B: IBZ periodic part -> remap
        wfs_ibz = ibzwfs._get_wfs(k_ibz, 0)
        grid_ibz = grid.new(kpt=wfs_ibz.psit_nX.desc.kpt_c,
                            dtype=wfs_ibz.psit_nX.desc.dtype)
        u_IBZ = wfs_ibz.psit_nX.ifft(grid=grid_ibz, periodic=True)

        N_c = np.array(u_IBZ.desc.size)
        r_c3xN = np.indices(N_c).reshape(3, -1)
        mapped = (U_cc.T @ r_c3xN
                  - (tau_c * N_c)[:, None]).astype(int)
        flat_idx = np.ravel_multi_index(mapped, N_c, mode='wrap')

        # Global phase from the translation: when transforming the
        # periodic part u(r) rather than the full Bloch function, a
        # constant phase exp(-/+i 2pi k_ibz . tau) appears.
        k_ibz_c = wfs_ibz.psit_nX.desc.kpt_c
        sign_tr = 1 if tr else -1
        global_phase = np.exp(sign_tr * 2j * np.pi * k_ibz_c @ tau_c)

        # Reciprocal lattice vector correction: the rotated k may
        # differ from k_BZ by a reciprocal lattice vector G.
        sign_k = -1 if tr else 1
        k_mapped_c = sign_k * U_cc @ k_ibz_c
        G_c = np.round(
            k_mapped_c - wfs_bz.psit_nX.desc.kpt_c).astype(int)

        nbands = min(2, u_IBZ.data.shape[0])
        for n in range(nbands):
            u_ref = u_IBZ.data[n].ravel()[flat_idx].reshape(N_c)
            if tr:
                u_ref = u_ref.conj()
            u_ref *= global_phase

            if G_c.any():
                r_frac = (np.indices(N_c).T / N_c).T
                G_phase = np.exp(2j * np.pi *
                                 np.einsum('c,c...->...', G_c, r_frac))
                u_ref = u_ref * G_phase

            assert np.allclose(u_BZ.data[n], u_ref, atol=1e-10), \
                f'K={K}, n={n}: max dev=' \
                f'{np.abs(u_BZ.data[n] - u_ref).max()}'
        tested += 1
    assert tested > 0, 'No non-symmorphic BZ k-points were tested'


# ---- Identity / equivalence tests ----

@pytest.mark.wannier
def test_bz_true_matches_bz_false_at_ibz_reps(si_calcs):
    """Under symmetry='off', bz=True and bz=False give identical output."""
    _require_new_stack(si_calcs['nosym'])

    calc = si_calcs['nosym']
    for K in range(len(calc.get_bz_k_points())):
        for n in range(4):
            a = calc.get_pseudo_wave_function(band=n, kpt=K, bz=False)
            b = calc.get_pseudo_wave_function(band=n, kpt=K, bz=True)
            assert np.allclose(a, b, atol=1e-12), \
                f'K={K}, n={n}: max diff={np.abs(a - b).max()}'


# ---- Caching / invalidation tests ----

@pytest.mark.wannier
def test_bz_wfs_P_ani_matches_independent_integration(si_calcs):
    """Lazy P_ani matches a fresh integration from the BZ psit_nX."""
    _require_new_stack(si_calcs['sym'])

    from gpaw.new.wannier import _bz_wfs

    calc = si_calcs['sym']
    ibzwfs = calc.dft.ibzwfs
    ibz = ibzwfs.ibz

    for K in range(len(ibz.bz)):
        wfs = _bz_wfs(ibzwfs, K, 0)
        P_cached = wfs.P_ani

        pt_aiG = wfs.psit_nX.desc.atom_centered_functions(
            [setup.pt_j for setup in wfs.setups],
            wfs.relpos_ac)
        P_fresh = pt_aiG.integrate(wfs.psit_nX)

        for a, P_ni in P_cached.items():
            assert np.allclose(P_ni, P_fresh[a], atol=1e-12), \
                f'K={K}, atom={a}: max dev=' \
                f'{np.abs(P_ni - P_fresh[a]).max()}'
