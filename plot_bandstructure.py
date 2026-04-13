from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from gpaw.new.ase_interface import GPAW

# --- Configuration ---
use_soc = True       # Set to False for scalar-relativistic bands
soc_bands = 30       # Number of bands to include in SOC (should match convergence)
use_cache = True     # If True, reuse existing bs.gpw instead of recomputing
spin = 'both'        # 'both', 'up' or 'down' -- selects spin character to plot.
                     # Linearly polarized light couples electrons/holes of the
                     # same spin, so filtering to 'up' or 'down' shows only
                     # bands relevant for singlet excitations.

# --- Band structure calculation (cached) ---
gs_gpw = 'gs_scs.gpw'
bs_gpw = 'bs.gpw'
if use_cache and Path(bs_gpw).is_file():
    calc = GPAW(bs_gpw)
else:
    calc = GPAW(gs_gpw).fixed_density(
        nbands=60,
        symmetry='off',
        kpts={'path': 'GKMG', 'npoints': 200},
        convergence={'bands': soc_bands, 'eigenvalues': 1e-6},
    )
    calc.write(bs_gpw, mode='all')  # mode='all' saves LCAO coefficients

# --- Get band structure path info ---
bs = calc.band_structure()
xcoords, label_xcoords, orig_labels = bs.get_labels()
label_xcoords = list(label_xcoords)

# --- LCAO setup (needed for layer projections in both cases) ---
ibzwfs = calc.dft.ibzwfs
setups = ibzwfs._wfs_u[0].setups
nao = setups.nao
nkpts = len(ibzwfs.ibz)

# Build boolean masks for basis functions on each layer
tags = calc.atoms.get_tags()
layer0_mask = np.zeros(nao, dtype=bool)
layer1_mask = np.zeros(nao, dtype=bool)
for a, tag in enumerate(tags):
    M = setups.M_a[a]
    M_slice = slice(M, M + setups[a].nao)
    if tag == 0:
        layer0_mask[M_slice] = True
    else:
        layer1_mask[M_slice] = True

if use_soc:
    # --- SOC eigenstates ---
    from gpaw.spinorbit import soc_eigenstates

    soc = soc_eigenstates(calc, n2=soc_bands)
    soc_eigs = soc.eigenvalues()      # (nkpts, 2*soc_bands), eV
    soc_vecs = soc.eigenvectors()     # (nkpts, 2*soc_bands, 2*soc_bands)
    fermi = soc.fermi_level           # eV
    nsoc = 2 * soc_bands

    # Compute SOC layer projections using LCAO coefficients + SOC eigenvectors.
    # Track each (spin, layer) Mulliken partial weight separately so we can
    # build normalized plot weights when filtering to a single spin channel.
    wup_L0_km = np.zeros((nkpts, nsoc))
    wup_L1_km = np.zeros((nkpts, nsoc))
    wdn_L0_km = np.zeros((nkpts, nsoc))
    wdn_L1_km = np.zeros((nkpts, nsoc))

    for wfs in ibzwfs:
        k = wfs.k
        C_nM = wfs.C_nM.gather(broadcast=True).data[:soc_bands]  # (soc_bands, nao)
        S_MM = wfs.S_MM.gather(broadcast=True).data               # (nao, nao)
        v_mn = soc_vecs[k]                                        # (nsoc, nsoc)

        # SOC state m = sum_j v_mn[m, 2j] * |j,up> + v_mn[m, 2j+1] * |j,dn>
        # For non-spin-polarized: C_jM is the same for both spins
        C_up_mM = v_mn[:, ::2] @ C_nM     # (nsoc, nao), spin-up component
        C_dn_mM = v_mn[:, 1::2] @ C_nM    # (nsoc, nao), spin-down component

        # Mulliken decomposition: w_nM = Re( C_nM* · sum_v S_{Mv} · C_nv )
        # which in matrix form is Re((C @ S.T) * C.conj()). Using S.T (not S)
        # is required at complex k-points; the two coincide only when S is
        # real-symmetric (Gamma). For Hermitian S this is equivalent to
        # Re((C @ S.conj()) * C.conj()).
        w_up = np.real((C_up_mM @ S_MM.T) * C_up_mM.conj())
        w_dn = np.real((C_dn_mM @ S_MM.T) * C_dn_mM.conj())

        wup_L0_km[k] = w_up[:, layer0_mask].sum(axis=1)
        wup_L1_km[k] = w_up[:, layer1_mask].sum(axis=1)
        wdn_L0_km[k] = w_dn[:, layer0_mask].sum(axis=1)
        wdn_L1_km[k] = w_dn[:, layer1_mask].sum(axis=1)

    # In parallel each kpt_comm rank only fills its own k-points; aggregate
    # across k-point groups so every rank has the full arrays before plotting
    # and before the normalization assertion.
    ibzwfs.kpt_comm.sum(wup_L0_km)
    ibzwfs.kpt_comm.sum(wup_L1_km)
    ibzwfs.kpt_comm.sum(wdn_L0_km)
    ibzwfs.kpt_comm.sum(wdn_L1_km)

    # Mulliken weights must sum to 1 for each SOC state: LCAO orthonormality
    # (C^dag S C = I) + unitarity of v_mn (rows are unit vectors from eigh)
    # guarantee this by construction. Assert loudly so a violation surfaces
    # instead of silently producing a misleading colormap.
    total_mk = wup_L0_km + wup_L1_km + wdn_L0_km + wdn_L1_km
    print(f'Mulliken weight sum: '
          f'min={total_mk.min():.6f}, max={total_mk.max():.6f}')
    assert np.allclose(total_mk, 1.0, atol=1e-6), (
        f'SOC Mulliken weights not normalized: '
        f'min={total_mk.min():.6f}, max={total_mk.max():.6f}'
    )

    # Build plot weights. For spin='both' the denominator is 1 (total Mulliken
    # weight) and we get the usual layer-0 fraction. For spin='up'/'down' we
    # normalize by the weight of the selected spin only -- this is the
    # conditional probability "given the state has the selected spin, what
    # fraction sits on layer 0?". The absolute weight of the selected spin is
    # kept in `spin_weight` and used as per-point alpha so bands dominated by
    # the other spin fade out of the plot.
    if spin == 'both':
        num = wup_L0_km + wdn_L0_km
        den = total_mk
    elif spin == 'up':
        num = wup_L0_km
        den = wup_L0_km + wup_L1_km
    elif spin == 'down':
        num = wdn_L0_km
        den = wdn_L0_km + wdn_L1_km
    else:
        raise ValueError(f"spin must be 'both', 'up' or 'down', got {spin!r}")

    # Safe division: where the selected spin has essentially zero weight, the
    # ratio is ill-defined; paint those points neutral (0.5) -- they'll be
    # invisible anyway thanks to the alpha.
    plot_weights = np.divide(num, den,
                             out=np.full_like(num, 0.5),
                             where=den > 1e-10)
    spin_weight = den  # used as alpha when spin != 'both'

    # Data for plotting
    plot_eigs = soc_eigs          # (nkpts, nsoc)
    plot_ref = fermi

else:
    # --- Scalar-relativistic bands ---
    energies = bs.energies  # (nspins, nkpts, nbands)
    nspins, _, nbands = energies.shape

    weights_layer0 = np.zeros((nspins, nkpts, nbands))
    total_skn = np.zeros((nspins, nkpts, nbands))

    for wfs in ibzwfs:
        s, k = wfs.spin, wfs.k
        C_nM = wfs.C_nM.gather(broadcast=True).data  # (nbands, nao)
        S_MM = wfs.S_MM.gather(broadcast=True).data   # (nao, nao)

        # Mulliken: w_nM = Re((C @ S.T) * C.conj()). The .T is needed at
        # complex k-points -- using plain S pairs the summed index wrongly
        # on the overlap and the sum only equals 1 when S is real-symmetric.
        CS_nM = C_nM @ S_MM.T
        w_nM = np.real(CS_nM * C_nM.conj())
        weights_layer0[s, k] = w_nM[:, layer0_mask].sum(axis=1)
        total_skn[s, k] = w_nM.sum(axis=1)

    # In parallel each kpt_comm rank only fills its own k-points; aggregate.
    ibzwfs.kpt_comm.sum(weights_layer0)
    ibzwfs.kpt_comm.sum(total_skn)

    # Mulliken weights must sum to 1 for each band (LCAO orthonormality).
    print(f'Mulliken weight sum: '
          f'min={total_skn.min():.6f}, max={total_skn.max():.6f}')
    assert np.allclose(total_skn, 1.0, atol=1e-6), (
        f'Mulliken weights not normalized: '
        f'min={total_skn.min():.6f}, max={total_skn.max():.6f}'
    )

    # Scalar-relativistic: 'up'/'down' map to spin channels 0/1 when present;
    # otherwise fall back to the single available channel.
    if spin == 'both' or nspins == 1:
        plot_eigs = energies[0]
        plot_weights = weights_layer0[0]
    elif spin == 'up':
        plot_eigs = energies[0]
        plot_weights = weights_layer0[0]
    elif spin == 'down':
        plot_eigs = energies[1]
        plot_weights = weights_layer0[1]
    else:
        raise ValueError(f"spin must be 'both', 'up' or 'down', got {spin!r}")
    spin_weight = np.ones_like(plot_weights)  # no fading in scalar case
    plot_ref = bs.reference

# --- Set up matplotlib figure ---
fig, ax = plt.subplots()


def pretty(kpt):
    if kpt == 'G':
        return r'$\Gamma$'
    elif len(kpt) == 2:
        return kpt[0] + '$_' + kpt[1] + '$'
    return kpt


labels = [pretty(name) for name in orig_labels]

# Merge adjacent special points at the same x-coordinate
i = 1
while i < len(labels):
    if label_xcoords[i - 1] == label_xcoords[i]:
        labels[i - 1] = labels[i - 1] + ',' + labels[i]
        labels.pop(i)
        label_xcoords.pop(i)
    else:
        i += 1

# Vertical lines at special k-points (skip endpoints)
for x in label_xcoords[1:-1]:
    ax.axvline(x, color='0.5')

# Reference energy (Fermi level)
ax.axhline(plot_ref, color='k', ls=':')

# VBM/CBM indicator lines.  With SOC every band holds one electron, so
# cbm_idx = N_e; without SOC bands are spin-degenerate and hold two, so
# cbm_idx = N_e // 2.
n_electrons = int(calc.get_number_of_electrons())
cbm_idx = n_electrons if use_soc else n_electrons // 2
vbm_idx = cbm_idx - 1
E_cbm = plot_eigs[:, cbm_idx].min()
E_vbm = plot_eigs[:, vbm_idx].max()
ax.axhline(E_cbm, color='k', ls=':')
ax.axhline(E_vbm, color='k', ls=':')

# Axis labels and limits
ax.set_xticks(label_xcoords)
ax.set_xticklabels(labels)
ax.set_ylabel('Energy [eV]')
ax.axis(xmin=0, xmax=xcoords[-1], ymin=-7, ymax=-3)

# --- Scatter plot colored by (normalized) layer 0 weight ---
nbands_plot = plot_eigs.shape[1]
X = np.tile(xcoords, (nbands_plot, 1)).T   # (nkpts, nbands_plot)

# When filtering to a single spin channel, fade bands of the opposite spin
# using `spin_weight` (the absolute Mulliken weight of the selected spin on
# each SOC state) as the per-point alpha.
if use_soc and spin != 'both':
    alpha = np.clip(spin_weight.ravel(), 0.0, 1.0)
    color_label = f'Layer 0 fraction (spin {spin})'
else:
    alpha = 1.0
    color_label = 'Layer 0 weight'

sc = ax.scatter(X.ravel(), plot_eigs.ravel(), c=plot_weights.ravel(),
                cmap='coolwarm', vmin=0, vmax=1, s=1, alpha=alpha)

fig.colorbar(sc, ax=ax, label=color_label)
bs_filename = f'bandstructure_spin_{spin}.png'
fig.savefig(bs_filename, dpi=200)
print(f'Saved {bs_filename}')
