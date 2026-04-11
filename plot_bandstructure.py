import numpy as np
import matplotlib.pyplot as plt
from gpaw.new.ase_interface import GPAW

# --- Configuration ---
use_soc = True    # Set to False for scalar-relativistic bands
soc_bands = 30    # Number of bands to include in SOC (should match convergence)

# --- Band structure calculation ---
gs_gpw = 'gs_scs.gpw'
calc = GPAW(gs_gpw).fixed_density(
    nbands=60,
    symmetry='off',
    kpts={'path': 'GKMG', 'npoints': 200},
    convergence={'bands': soc_bands},
)
calc.write('bs.gpw', mode='all')  # mode='all' saves LCAO coefficients

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

    # Compute SOC layer projections using LCAO coefficients + SOC eigenvectors
    weights_layer0 = np.zeros((nkpts, nsoc))

    for wfs in ibzwfs:
        k = wfs.k
        C_nM = wfs.C_nM.gather(broadcast=True).data[:soc_bands]  # (soc_bands, nao)
        S_MM = wfs.S_MM.gather(broadcast=True).data               # (nao, nao)
        v_mn = soc_vecs[k]                                        # (nsoc, nsoc)

        # SOC state m = sum_j v_mn[m, 2j] * |j,up> + v_mn[m, 2j+1] * |j,dn>
        # For non-spin-polarized: C_jM is the same for both spins
        C_up_mM = v_mn[:, ::2] @ C_nM     # (nsoc, nao), spin-up component
        C_dn_mM = v_mn[:, 1::2] @ C_nM    # (nsoc, nao), spin-down component

        # Mulliken decomposition with both spin components
        w_up = np.real((C_up_mM @ S_MM) * C_up_mM.conj())
        w_dn = np.real((C_dn_mM @ S_MM) * C_dn_mM.conj())
        w_mM = w_up + w_dn

        weights_layer0[k] = w_mM[:, layer0_mask].sum(axis=1)

    # Sanity check
    total = w_mM.sum(axis=1)
    print(f'Mulliken weight sum (last k): min={total.min():.6f}, max={total.max():.6f}')

    # Data for plotting
    plot_eigs = soc_eigs         # (nkpts, nsoc)
    plot_weights = weights_layer0  # (nkpts, nsoc)
    plot_ref = fermi

else:
    # --- Scalar-relativistic bands ---
    energies = bs.energies  # (nspins, nkpts, nbands)
    nspins, _, nbands = energies.shape

    weights_layer0 = np.zeros((nspins, nkpts, nbands))

    for wfs in ibzwfs:
        s, k = wfs.spin, wfs.k
        C_nM = wfs.C_nM.gather(broadcast=True).data  # (nbands, nao)
        S_MM = wfs.S_MM.gather(broadcast=True).data   # (nao, nao)

        CS_nM = C_nM @ S_MM
        w_nM = np.real(CS_nM * C_nM.conj())
        weights_layer0[s, k] = w_nM[:, layer0_mask].sum(axis=1)

    # Data for plotting (use first spin channel)
    plot_eigs = energies[0]         # (nkpts, nbands)
    plot_weights = weights_layer0[0]  # (nkpts, nbands)
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

# Axis labels and limits
ax.set_xticks(label_xcoords)
ax.set_xticklabels(labels)
ax.set_ylabel('Energy [eV]')
ax.axis(xmin=0, xmax=xcoords[-1], ymin=-7, ymax=-3)

# --- Scatter plot colored by layer 0 weight ---
nbands_plot = plot_eigs.shape[1]
X = np.tile(xcoords, (nbands_plot, 1)).T   # (nkpts, nbands_plot)

sc = ax.scatter(X.ravel(), plot_eigs.ravel(), c=plot_weights.ravel(),
                cmap='coolwarm', vmin=0, vmax=1, s=1)

fig.colorbar(sc, ax=ax, label='Layer 0 weight')
fig.savefig('bandstructure.png', dpi=200)
print('Saved bandstructure.png')
