# Phonon Kernel BSE Pipeline - Bug Analysis

## BUG 1 (Critical): q-point sign convention mismatch

**Files**: `q_list.py:12` vs `gmatrix.py:307`

`q_list.py` computes `q = k_i - k_j`, but `ElectronPhononMatrix.bloch_matrix`
uses the convention that q is the phonon wavevector such that the bra state is
at `k + q`. For the coupling `<m, k'| ... |n, k>` one needs `q = k' - k`.

In `construct_Kph`, `g_unique_q[:, q_indices[k, kp], k, ...]` fetches g
with `q = k - kp` (from q_list.py), but this gives `<m, k+(k-kp)|...|n, k> =
<m, 2k-kp|...|n, k>` instead of `<m, kp|...|n, k>`.

**Fix**: Change `q_list.py:12` to `q_diff = kpts[None, :, :] - kpts[:, None, :]`
(i.e., `q = k_j - k_i`), or swap to `q_indices[kp, k]` when accessing g in
`construct_Kph`.

## BUG 2 (Critical): Wrong effective mass in prefactor

**File**: `bse.py:1175-1181`

The code uses `total_mass_amu * amu/me` as the effective mass, but
`_bloch_matrix` in `gmatrix.py:226-228` uses just `amu/me` (a unit conversion
constant). The phonon eigenvectors are already mass-scaled, so the mass is
absorbed. Using total cell mass introduces a spurious factor of
`1/sqrt(total_mass_amu)`.

**Fix**: Replace `mass_factor = total_mass_amu * (amu / me)` with
`mass_factor = amu / me`, and adjust the frequency units to match
`_bloch_matrix`:
```python
scale_factor = 1.0 / np.sqrt(2 * (amu / me) / units.Hartree * w_l_safe)
```

## BUG 3 (Critical): Unit mismatch in g-matrix scaling

**Files**: `g.py:36-37`, `bse.py:1184-1188`

`g.py` calls `bloch_matrix(..., prefactor=False)` which returns g in eV/Ang.
`construct_Kph` then multiplies by an atomic-unit prefactor and divides by
Hartree, mixing unit systems. The g-matrix should first be converted to
atomic units (Hartree/Bohr) before applying the atomic-unit prefactor.

**Fix**: Convert g from eV/Ang to Hartree/Bohr first:
```python
g_au = g_eVAng / (units.Hartree / units.Bohr)
g_full_au = g_au / np.sqrt(2 * amu/me / units.Hartree * w_l_safe)
```

## BUG 4 (Critical): Einsum produces 1D array instead of 2D kernel matrix

**File**: `bse.py:1267-1271`

The einsum for `M1_wt` uses the same index `t` for both the bra and ket
BSE eigenstate, producing a 1D array. The phonon kernel should be a matrix
`K_ph(t, t')` with two independent eigenstate indices.

**Fix**: Use separate indices:
```python
M1_ut = np.einsum('vcu, slca, slvb, bat, bctl -> ut',
                  Ap_vcu, g_slcc, g_slvv, A_vct, nkkp_vcwl)
```

## BUG 5 (Moderate): C_knm band index mapping is fragile

**File**: `phases_kpq.py:22-23`, `bse.py:1160-1164`

Hard-coded band ranges `range(3,15)` and `range(1,30)` make the C-matrix
indexing fragile. The slicing `C[:nv, :nv]` for valence and `C[nv:nv+nc,
nv:nv+nc]` for conduction assumes a specific alignment between LCAO, PW,
and BSE band orderings that may not hold.

## BUG 6 (Moderate): g-matrix band slicing may not match intended bands

**File**: `bse.py:1167-1168`

`g[:, :, :, :, :nv, :nv]` assumes the first `nv` LCAO bands are valence,
but LCAO band ordering depends on the calculation setup and may not
match the BSE valence band range.

## BUG 7 (Minor): g.py chunk concatenation skips cached chunks

**File**: `g.py:27-54`

If previously computed chunks exist, they are skipped but not loaded,
so `all_gs` is incomplete. Should load all chunks at the end regardless.

## BUG 8 (Minor): Phonon calculator inconsistency

**File**: `phonon_energies.py:12-13`

Phonon frequencies computed with PW calculator but elph coupling
computed with LCAO calculator. Should use consistent setup.
