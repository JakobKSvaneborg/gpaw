"""Test symmetry-unfolding of wave functions from the IBZ to the full BZ.

This verifies that ``get_bz_pseudo_wave_function`` (new GPAW) returns
the same wave function (up to a global phase) as a separate calculation
run with ``symmetry='off'``.
"""
import numpy as np
import pytest
from ase.build import bulk

from gpaw import GPAW, PW


def _phase_aligned_overlap(psi_a, psi_b):
    """Return |<a|b>| / (||a|| ||b||).

    For two pseudo wave functions representing the same Bloch state
    (possibly up to a global phase), this should equal 1.
    """
    inner = np.vdot(psi_a, psi_b)
    norm_a = np.linalg.norm(psi_a)
    norm_b = np.linalg.norm(psi_b)
    return abs(inner) / (norm_a * norm_b)


@pytest.mark.serial
def test_bz_unfolding_matches_nosym(in_tmp_dir, gpaw_new):
    """Unfolded wfs at non-IBZ k-points must match a nosym reference.

    Runs bulk Si once with symmetry enabled and once with symmetry off.
    At a k-point that is mapped out of the IBZ by a point-group
    operation, the wave function obtained by symmetry-unfolding should
    equal the reference (up to a global phase) for a non-degenerate
    band.
    """
    if not gpaw_new:
        pytest.skip('get_bz_pseudo_wave_function only exists in new GPAW')

    atoms = bulk('Si')

    common = dict(
        mode=PW(200),
        kpts={'size': (2, 2, 2), 'gamma': True},
        nbands=8,
        convergence={'density': 1e-6},
        txt=None,
    )

    atoms_sym = atoms.copy()
    atoms_sym.calc = GPAW(**common)
    atoms_sym.get_potential_energy()
    calc_sym = atoms_sym.calc

    atoms_nosym = atoms.copy()
    atoms_nosym.calc = GPAW(symmetry='off', **common)
    atoms_nosym.get_potential_energy()
    calc_nosym = atoms_nosym.calc

    # Sanity: BZ k-points must be listed in the same order, because
    # the BZ index semantics are what we are comparing.
    bz_sym = calc_sym.get_bz_k_points()
    bz_nosym = calc_nosym.get_bz_k_points()
    assert np.allclose(bz_sym, bz_nosym)

    # Symmetry reduction should be non-trivial here.
    assert len(calc_sym.get_ibz_k_points()) < len(bz_sym)

    # Find a non-IBZ k-point (i.e. one that needs an actual transform).
    bz2ibz = calc_sym.get_bz_to_ibz_map()
    ibz = calc_sym.dft.ibzwfs.ibz
    non_ibz_K = None
    for K in range(len(bz_sym)):
        s_op = int(ibz.s_K[K])
        tr = bool(ibz.time_reversal_K[K])
        if not (s_op == 0 and not tr):
            non_ibz_K = K
            break
    assert non_ibz_K is not None, 'No BZ k-point needs a real transform'

    # Compare wavefunctions band-by-band.  Restrict to non-degenerate
    # bands since the unfolded and nosym wfs may differ by a rotation
    # within any degenerate subspace.
    eps_nosym = calc_nosym.get_eigenvalues(kpt=non_ibz_K)
    # Match the symmetry eigenvalues (expanded from IBZ) with nosym:
    ibz_kpt = bz2ibz[non_ibz_K]
    eps_sym = calc_sym.get_eigenvalues(kpt=ibz_kpt)
    assert np.allclose(eps_sym, eps_nosym, atol=1e-4)

    min_gap = 0.05  # eV
    checked = 0
    for n in range(len(eps_sym)):
        # Skip degenerate bands.
        neighbours = [e for i, e in enumerate(eps_sym) if i != n]
        if min(abs(e - eps_sym[n]) for e in neighbours) < min_gap:
            continue
        psi_sym = calc_sym.get_bz_pseudo_wave_function(
            band=n, kpt=non_ibz_K, spin=0)
        psi_nosym = calc_nosym.get_pseudo_wave_function(
            band=n, kpt=non_ibz_K, spin=0)
        overlap = _phase_aligned_overlap(psi_sym, psi_nosym)
        assert overlap == pytest.approx(1.0, abs=1e-3), (
            f'band {n} at BZ k-point {non_ibz_K}: |<sym|nosym>| = '
            f'{overlap}')
        checked += 1

    assert checked > 0, 'No non-degenerate bands to compare'


@pytest.mark.serial
def test_bz_unfolding_identity_matches_get_pseudo(in_tmp_dir, gpaw_new):
    """For a BZ k-point that IS the IBZ representative (identity),
    ``get_bz_pseudo_wave_function`` must return exactly the same thing
    as ``get_pseudo_wave_function``.
    """
    if not gpaw_new:
        pytest.skip('get_bz_pseudo_wave_function only exists in new GPAW')

    atoms = bulk('Si')
    atoms.calc = GPAW(
        mode=PW(200),
        kpts={'size': (2, 2, 2), 'gamma': True},
        nbands=8,
        convergence={'density': 1e-6},
        txt=None,
    )
    atoms.get_potential_energy()
    calc = atoms.calc

    ibz = calc.dft.ibzwfs.ibz
    # Find a BZ k-point that coincides with its IBZ representative.
    identity_K = None
    for K in range(len(calc.get_bz_k_points())):
        if int(ibz.s_K[K]) == 0 and not bool(ibz.time_reversal_K[K]):
            identity_K = K
            break
    assert identity_K is not None

    ibz_kpt = int(ibz.bz2ibz_K[identity_K])
    for n in range(4):
        psi_bz = calc.get_bz_pseudo_wave_function(
            band=n, kpt=identity_K, spin=0)
        psi_ibz = calc.get_pseudo_wave_function(
            band=n, kpt=ibz_kpt, spin=0)
        assert np.allclose(psi_bz, psi_ibz)
